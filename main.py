import json
import os
import sys
import datetime
from setproctitle import setproctitle
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, load_from_disk

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from ATE.system_prompts import *

setproctitle('yongchan')

def preprocess_data(x, tokenizer, model_name):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(*DOMAIN_TO_PROMPT[x['domain']])
    user_prompt = x['text']
    
    if 'llama' in model_name:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    elif 'gemma' in model_name:
        combined_prompt = f"{system_prompt}\n{user_prompt}"
        messages = [{"role": "user", "content": combined_prompt}]
        
    label_str = ''
    if len(x['label'])==0:
        label_str = 'No terminology'
    else:
        for i, label_element in enumerate(x['label']):
            if i != len(x['label'])-1:
                label_str += label_element + ', '
            else:
                label_str += label_element
    
    x['input_ids'] = tokenizer.apply_chat_template(messages, add_generation_prompt=True)['input_ids']
    x['labels'] = tokenizer(label_str)
    return x

from typing import List, Dict, Any

def f1_score(
    pred: List[str], 
    ref: List[str], 
    )->Dict[str, int]:
    individual_result = {}  
    
    tp, len_pred, len_ref = 0, len(pred), len(ref)
    for relation in pred:
        if relation in ref:
            tp += 1
    
    precision = tp / len_pred
    recall = tp / len_ref

    if not precision and not recall:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    individual_result['tp'] = tp
    individual_result['len_pred'] = len_pred
    individual_result['len_ref'] = len_ref
    individual_result['precision'] = precision
    individual_result['recall'] = recall
    individual_result['f1_score'] = f1_score

    return individual_result

def micro_f1_score(
    total_tp: int, 
    total_len_pred: int, 
    total_len_ref: int, 
    )->Dict[str, Any]:
    overall_result = {}
    
    precision = total_tp / total_len_pred if total_len_pred > 0 else 0
    recall = total_tp / total_len_ref if total_len_ref > 0 else 0

    if not precision and not recall:
        micro_f1_score = 0
    else:
        micro_f1_score = 2 * precision * recall / (precision + recall)

    overall_result["total_precision"] = precision
    overall_result["total_recall"] = recall
    overall_result["micro_f1_score"] = micro_f1_score

    return overall_result


def eval_metric(x, tokenizer, report_save_path, metric1, metric2):
    import pandas as pd
    input_ids = x.inputs
    preds = tokenizer.batch_decode(x.predictions[input_ids.shape[-1]:], skip_special_tokens=True)
    refs = tokenizer.batch_decode(x.label_ids[input_ids.shape[-1]:], skip_special_tokens=True)
    input_ids
    
    f1_scores, precisions, recalls = [], [], []
    total_tp, total_len_pred, total_len_ref = 0, 0, 0
    for pred, ref in zip(preds, refs):
        individual_score = metric1(pred, ref)
        total_tp += individual_score['tp']
        total_len_pred += individual_score['len_pred']
        total_len_ref += individual_score['len_ref']
        
        f1_scores.append(individual_score['f1_score'])
        precisions.append(individual_score['precision'])
        recalls.append(individual_score['recall'])
    
    overall_result = metric2(total_tp, total_len_pred, total_len_ref)

    individual_report = pd.DataFrame({
        'prediction': preds,
        'reference': refs,
        'f1_score': f1_scores,
        'precision': precisions,
        'recall': recalls
    })
    
    reports = os.listdir(os.path.dirname(report_save_path))
    if len(reports):
        last_report_num = sorted(reports, key=lambda x: int(os.path.splitext(x)[0].split('_'))[-1], reverse=True)[0]
        report_save_path = report_save_path.format(last_report_num)
    else:
        report_save_path = report_save_path.format(0)
        
    individual_report.to_csv(report_save_path, index=False)
    
    overall_report = {
        'total_precision': overall_result['total_precision'],
        'total_recall': overall_result['total_recall'],
        'micro_f1_score': overall_result['micro_f1_score']
    }
    
    return overall_report
    
    
def main(cfg):
    dataset_name = cfg['dataset_name']
    dataset_path = cfg['dataset_path']
    train_mode = cfg['train_mode']
    model_name = cfg['model_name']
    output_dir = cfg['output_dir']
    
    if dataset_name == 'ACTER' and not dataset_path:
        raise ValueError('Dataset path is required for ACTER dataset')
    
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    output_dir = os.path.join(now, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    report_save_path = os.path.join(output_dir, 'report', 'report_{0}.csv')
    os.makedirs(report_save_path, exist_ok=True)
    
    if dataset_name == 'ACTER':
        dataset = load_from_disk(dataset_path)
        dataset_pre_func = lambda x: preprocess_data(x, tokenizer, model_name)
        cache_file_names = [os.path.join(f"{dataset_path}", "cache", f"{mode}_setup.cache") for mode in ['train', 'validation', 'test']]
        
    dataset = dataset.map(dataset_pre_func, num_proc=8, cache_file_names=cache_file_names)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=4, return_tensors='pt', padding=True)
    compute_metrics = lambda x, tokenizer, report_save_path, f1_score, micro_f1_score: eval_metric(x, tokenizer, report_save_path, f1_score, micro_f1_score)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        warmup_steps=cfg['warmup_steps'],
        gradient_checkpointing=False,
        fp16=True,
        evaluation_strategy=cfg['evaluation_strategy'],
        dataloader_num_workers=cfg['dataloader_num_workers'],
        generation_max_length=cfg['generation_max_length'],
        per_device_eval_batch_size=cfg['per_device_eval_batch_size'],
        eval_steps=cfg['eval_steps'],
        save_steps=cfg['save_steps'],
        logging_steps=cfg['logging_steps'],
        max_new_tokens=512,
        num_train_epochs=cfg['num_train_epochs'],
        load_best_model_at_end=False,
        metric_for_best_model='micro_f1_score',
        greater_is_better=True,
        save_total_limit=cfg['save_total_limit'],
        push_to_hub=False,
        label_names=['label'],
        include_inputs_for_metrics=True,
    )

    trainer = Seq2SeqTrainer(
        mdoel=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'])

    if train_mode:
        trainer.eval_dataset=dataset['validation']
        trainer.train()
    else:
        trainer.eval_dataset=dataset['test']
        trainer.evaluate()
        
if __name__ == '__main__':
    with open('config.json') as f:
        cfg = json.load(f)
    main(cfg)