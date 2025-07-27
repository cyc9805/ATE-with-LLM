import json
import os
import re
import logging
import torch

from ast import literal_eval
from rich.logging import RichHandler
from datetime import datetime
from argparse import ArgumentParser, ArgumentTypeError
from transformers import Seq2SeqTrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_from_disk
from llama_index.core import StorageContext, load_index_from_storage
from prompt_template import *
from vector_store import CustomVectorStore, VectorStoreQueryMode
from trainer import ATETrainer
from data import preprocess_data, initialize_gpt, create_retriever
from utils import f1_score, micro_f1_score

DATASET_NAME_TO_PATH = {
    'ACTER': 'dataset/ACTER/huggingface',
    'ACL-RD': 'dataset/ACL-RD/huggingface',
    'BCGM': 'dataset/BCGM/huggingface'
}

DATASET_NAME_TO_PERSIST_DIR = {
    'ACTER': 'dataset/ACTER/index',
    'ACL-RD': 'dataset/ACL-RD/index',
    'BCGM': 'dataset/BCGM/index'
}

os.environ["WANDB_PROJECT"] = "Automatic-Term-Extraction"

def preprocess_text(input_text, text, delimiter=', ', is_pred=True):
    if is_pred:
        if "let me know" in text.lower() and (text.lower().endswith("! \n") or text.lower().endswith("!\n")):
            text = text[:text.lower().index("let me know")]

        if "## reason:" in text.lower():
            text = text[:text.lower().find("## reason:")].strip()
        
        if "**explanation:**" in text.lower():
            text = text[:text.lower().find("**explanation:**")].strip()
            
        while text.startswith('\n'):
            text = text[1:]
            
        while text.endswith('\n'):
            text = text[:-1]
            
        if '\n' in text:
            text = text.split('\n')[-1].strip()
            
        lowered_text = text.lower()
        lowered_input_text = input_text.lower()
        
        keywords1 = ['answer:', 'output:', 'are:', 'is:' 'terms:', 'domain:', 'terms**', 'output**', '**output:**']
        keywords2 = ['terms are', 'term is', 'should be:', 'would be:', 'answer is:', 'answer is ', 'output is:', 'the annotator should write']
        for keyword in keywords1+keywords2:
            if keyword in lowered_text:
                if keyword in keywords2 and lowered_text not in lowered_input_text:
                    text = text[lowered_text.find(keyword)+len(keyword):].strip()
                else:
                    text = text[lowered_text.rfind(keyword)+len(keyword):].strip()
                break
        
        # Eliminate any special characters at the prefix
        pattern1 = r'^[^\w]+'
        text = re.sub(pattern1, ' ', text)
    
    # Eliminate parantheses
    pattern2 = r'[\[\]\"\']'
    text = re.sub(pattern2, ' ', text)
        
    domain_words = []
    for domain_word in text.split(delimiter):
        while domain_word.strip().startswith('.') or domain_word.strip().endswith('.'):
            domain_word = domain_word.replace('.', ' ')
        domain_words.append(domain_word)
    
    if is_pred:
        splited_domain_words = []
        for domain_word in domain_words:
            lowered_domain_word = domain_word.lower()
            if lowered_domain_word.startswith('and ') or ' and ' in lowered_domain_word and lowered_domain_word.strip() not in lowered_input_text:
                domain_words.remove(domain_word)
                for splited_domain_word in domain_word.split('and '):
                    splited_domain_word = splited_domain_word.strip()
                    if splited_domain_word != '':
                        splited_domain_words.append(splited_domain_word)
        
        domain_words.extend(splited_domain_words)
        
        cnt_threshold = 50
        cnt = 0
        for i in range(len(domain_words)):
            for j in range(i+1, len(domain_words)):
                if domain_words[i] == domain_words[j]:
                    cnt += 1
                if cnt > cnt_threshold:
                    domain_words = list(set(domain_words))
                    print('Repetitive answers are removed:', domain_words)
                    break
            if cnt > cnt_threshold:
                break
            
    domain_words = list(map(lambda x: x.strip(), domain_words))
    return domain_words   

    
def eval_metric(x, tokenizer, report_save_path, metric1, metric2, preprocessor, is_gpt_model=False, is_enc_dec_model=False):
    import pandas as pd
    
    if not is_gpt_model:
        preprocessed_preds = []
        preprocessed_refs = []
    
        batched_input_ids = x.inputs
        batched_preds_tokenized = x.predictions
        batched_refs_tokenized = x.label_ids
        
        input_texts, preds, refs = [], [], []

        for preds_tokenized, refs_tokenized, input_ids in zip(batched_preds_tokenized, batched_refs_tokenized, batched_input_ids):
            input_ids = [i for i in input_ids if i != -100]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            input_texts.append(input_text)
            
            pred = tokenizer.decode([i for i in preds_tokenized[0 if is_enc_dec_model else len(input_ids):] if i != -100], skip_special_tokens=True)
            preprocessed_pred = preprocessor(input_text, pred, True)
            preds.append(pred)
            preprocessed_preds.append(preprocessed_pred)
            
            ref = tokenizer.decode([i for i in refs_tokenized if i != -100], skip_special_tokens=True)
            preprocessed_ref = preprocessor(input_text, ref, False)
            refs.append(ref)
            preprocessed_refs.append(preprocessed_ref)
    else:
        input_texts = x['text']
        preds = x['predictions']
        refs = x['labels']
        preprocessed_refs = [preprocessor(input_text, ref, False) for input_text, ref in zip(input_texts, refs)]
        preprocessed_preds = [preprocessor(input_text, pred, True) for input_text, pred in  zip(input_texts, preds)]
        
    f1_scores, precisions, recalls = [], [], []
    total_tp, total_len_pred, total_len_ref = 0, 0, 0
    for pred, ref in zip(preprocessed_preds, preprocessed_refs):
        individual_score = metric1(pred, ref)
        total_tp += individual_score['tp']
        total_len_pred += individual_score['len_pred']
        total_len_ref += individual_score['len_ref']
        
        f1_scores.append(individual_score['f1_score'])
        precisions.append(individual_score['precision'])
        recalls.append(individual_score['recall'])
    
    overall_result = metric2(total_tp, total_len_pred, total_len_ref)

    individual_report = pd.DataFrame({
        'input_text': input_texts,
        'prediction': preds,
        'normalized_prediction': preprocessed_preds,
        'reference': refs,
        'normalized_reference': preprocessed_refs,
        'f1_score': f1_scores,
        'precision': precisions,
        'recall': recalls
    })
    
    reports = os.listdir(os.path.dirname(report_save_path))
    if len(reports):
        if 'report_overall.csv' in reports:
            reports.remove('report_overall.csv') 
        last_report_num = max(list(map(lambda x: int(os.path.splitext(x)[0].split('_')[-1]), reports)))
        individual_report_save_path = report_save_path.format(last_report_num+1)
    else:
        individual_report_save_path = report_save_path.format(0)
        
    individual_report.to_csv(individual_report_save_path, index=False)
    
    overall_report = {
        'total_precision': overall_result['total_precision'],
        'total_recall': overall_result['total_recall'],
        'micro_f1_score': overall_result['micro_f1_score']
    }
    
    with open(report_save_path.format('overall'), 'w') as f:
        f.write(json.dumps(overall_report, ensure_ascii=False, indent=4))
        
    return overall_report
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def main(cfg):
    dataset_name = cfg['dataset_name']
    num_shots = cfg['num_shots']
    train_mode = cfg['train_mode']
    model_name = cfg['model_name']
    output_dir = cfg['output_dir']
    bf16 = cfg['bf16']
    checkpoint_path = cfg['checkpoint_path']
    label_delimiter = cfg['label_delimiter']
    retrieval_method = cfg['retrieval_method']
    temperature = cfg['temperature']
    do_sample = cfg['do_sample']
    seed = cfg['seed']
    
    retrieval_methods = list(VectorStoreQueryMode.__members__.values())
    if retrieval_method not in retrieval_methods:
        raise ValueError(f'Retrieval method should be one of {retrieval_methods}')
    
    if 'bart' in model_name and cfg['generation_max_length'] > 1024:
        logging.warning('Bart model does not support generation_max_length > 1024. Setting generation_max_length to 1024')
        cfg['generation_max_length']=1024
    
    dataset_path = DATASET_NAME_TO_PATH[dataset_name]
    
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    if 'bart' in model_name:
        output_dir = os.path.join(output_dir, 'train' if train_mode else 'test', dataset_name, 'bart', now)
    else:
        output_dir = os.path.join(output_dir, 'train' if train_mode else 'test', dataset_name, retrieval_method, now)
    
    report_save_path = os.path.join(output_dir, 'report', 'report_{0}.csv')
    os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=4))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, "run.log")),
                  RichHandler()])

    pretrained_model = checkpoint_path if checkpoint_path else model_name
    
    if 'gpt' in model_name:
        initialize_gpt()
        tokenizer = None
        split_dataset_types = ['test']
        
    elif 'bart' in model_name:
        from ATE.models import BartForATE
        from transformers import BartTokenizerFast

        model = BartForATE.from_pretrained(pretrained_model)
        tokenizer = BartTokenizerFast.from_pretrained(pretrained_model, src_lang="en_XX", tgt_lang="en_XX")
        model.config.dropout = cfg.get('dropout', 0.3)
        model.config.attention_dropout=cfg.get('attention_dropout', 0.1)
        model.config.label_smoothing=cfg.get('label_smoothing', 0.2)
        split_dataset_types = ['train', 'validation', 'test']
        is_encoder_decoder = True
    else:
        logging.info(f"Loading model: {pretrained_model}")
        kwargs = {}
        if 'gemma' in model_name:
            kwargs = {"attn_implementation": "eager"}
            
        model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16 if bf16 else torch.float16, device_map='auto', **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        tokenizer.padding_side = 'left'
        if 'llama' in model_name:
            tokenizer.pad_token=tokenizer.eos_token
        if 'mistral' in model_name:
            tokenizer.pad_token=tokenizer.eos_token
        split_dataset_types = ['test']
        is_encoder_decoder = False

    index, retriever = None, None
    num_proc=1
    if 'bart' not in model_name:
        delimiter = label_delimiter         
        if retrieval_method == "default_w_ins":
            embed_model = "local:BAAI/bge-en-icl"
        else:
            embed_model = "local:BAAI/bge-large-en-v1.5"

        if retrieval_method in ["fastkassim"]:
            num_proc=60
        
        persist_dir = DATASET_NAME_TO_PERSIST_DIR[dataset_name]
        persist_dir = os.path.join(persist_dir, os.path.basename(embed_model))    
        retrieved_cache_dir = os.path.join(persist_dir, retrieval_method, str(seed) if retrieval_method == "random" else "")
        retrieved_cache_name = "retrieved_result_{0}.json"
        retrieved_cache_path = os.path.join(retrieved_cache_dir, retrieved_cache_name)
        os.makedirs(retrieved_cache_dir, exist_ok=True)
        cache_files = os.listdir(retrieved_cache_dir)
        max_cache_file_num  = None
        if len(cache_files) > 0:
            max_cache_file_num = max(list(map(lambda x: int(os.path.splitext(x.split('_')[-1])[0]), cache_files)))
        if max_cache_file_num is not None and max_cache_file_num >= num_shots:
            retrieved_cache_path = retrieved_cache_path.format(max_cache_file_num)
            logging.info(f"Loading previously retrieved results from {retrieved_cache_path}")
        elif num_shots > 0:
            logging.info(f"Loading storage context from {persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            logging.info(f"Loading index from storage")
            index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
            index._vector_store = CustomVectorStore(index._vector_store.data) 
            retriever = create_retriever(index, dataset_name, num_shots, retrieval_method, seed)

    else:
        delimiter = ' ; '
        retrieved_cache_path=None

    preprocessor = lambda input_text, x, is_pred: preprocess_text(input_text, x, delimiter, is_pred)            
    dataset = load_from_disk(dataset_path)
    
    cache_file_names = {
        mode: os.path.join(
            f"{dataset_path}", 
            model_name, 
            str(num_shots), 
            retrieval_method, 
            str(seed) if retrieval_method == 'random' else '',
            f"{mode}_setup.cache") for mode in ['train', 'validation', 'test']}
    
    for dataset_type in split_dataset_types:
        cache_file_name = cache_file_names[dataset_type]
        os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
        is_train_dataset = dataset_type == 'train'
        dataset_pre_func = lambda x, i: preprocess_data(x, i, tokenizer, model_name, dataset_name, retriever, retrieved_cache_path, num_shots, is_train_dataset, retrieval_method, delimiter, seed)    
        dataset[dataset_type] = dataset[dataset_type].map(dataset_pre_func, num_proc=num_proc, cache_file_name=cache_file_names[dataset_type], with_indices=True)
    
    # If there are any retrieved results, save them
    if 'test' in split_dataset_types and 'retrieved_result' in dataset['test'].column_names:
        total_retrieved_results = dataset['test']['retrieved_result']
        texts = dataset['test']
        total_retrieved_results =   {i: retrieved_result for text, (i, retrieved_result) in zip(texts, enumerate(list(map(lambda x: literal_eval(x), total_retrieved_results))))}
        retrieved_cache_path = os.path.join(retrieved_cache_dir, retrieved_cache_name.format(str(num_shots)))
        with open(retrieved_cache_path, 'w') as f:
            json.dump(total_retrieved_results, f, ensure_ascii=False, indent=4)
        dataset[dataset_type] = dataset[dataset_type].remove_columns('retrieved_result')
        
    # Clear out unnecessary index and retriever to save GPU memory
    del index, retriever

    if 'gpt' in model_name:
        for dataset_type in split_dataset_types:
            eval_metric(dataset[dataset_type], tokenizer, report_save_path, f1_score, micro_f1_score, preprocessor, True)
        return
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=4, return_tensors='pt', padding=True)
    
    compute_metrics = lambda x: eval_metric(x, tokenizer, report_save_path, f1_score, micro_f1_score, preprocessor, False, is_encoder_decoder)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],
        adam_beta1=cfg['adam_beta1'],
        adam_beta2=cfg['adam_beta2'],
        adam_epsilon=cfg['adam_epsilon'],
        weight_decay=cfg['weight_decay'],
        warmup_steps=cfg['warmup_steps'],
        gradient_checkpointing=False,
        bf16=bf16,
        fp16=not bf16,
        evaluation_strategy=cfg['evaluation_strategy'],
        dataloader_num_workers=cfg['dataloader_num_workers'],
        generation_max_length=cfg['generation_max_length'],
        generation_num_beams=cfg['generation_num_beams'],
        per_device_eval_batch_size=cfg['per_device_eval_batch_size'],
        eval_steps=cfg['eval_steps'],
        save_steps=cfg['save_steps'],
        logging_steps=cfg['logging_steps'],
        num_train_epochs=cfg['num_train_epochs'],
        load_best_model_at_end=False,
        metric_for_best_model='micro_f1_score',
        greater_is_better=True,
        save_total_limit=cfg['save_total_limit'],
        push_to_hub=False,
        label_names=['labels'],
        include_inputs_for_metrics=True,
        predict_with_generate=True,
        do_predict=True,
        auto_find_batch_size=cfg['auto_find_batch_size'],
        resume_from_checkpoint=checkpoint_path,
    )

    trainer = ATETrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'])     # We modify the eval_dataset to later depending on the mode

    gen_kwargs = {
        "temperature": temperature,
        "use_cache": True,
        "do_sample": do_sample,
        # "top_p": 0.9,
        # "num_beams":1,
    }
    
    if train_mode:
        logging.info('Start training')
        trainer.eval_dataset=dataset['validation']
        trainer.train()
    else:
        logging.info('Start evaluation')
        trainer.eval_dataset=dataset['test']
        trainer.evaluate(**gen_kwargs)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/test.json')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='ACTER')
    parser.add_argument('--num_shots', type=str, default='0')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--label_delimiter', type=str, default=', ')
    parser.add_argument('--retrieval_method', type=str, default='random')
    parser.add_argument('--temperature', type=str, default='0.01')
    parser.add_argument('--do_sample', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    config_path = args.config_path
    model_name = args.model_name
    dataset_name = args.dataset_name
    num_shots = args.num_shots
    checkpoint_path = args.checkpoint_path
    label_delimiter = args.label_delimiter
    retrieval_method = args.retrieval_method
    temperature = args.temperature
    do_sample = args.do_sample
    seed = args.seed

    with open(config_path) as f:
        cfg = json.load(f)
    
    cfg['model_name'] = model_name
    cfg['dataset_name'] = dataset_name
    cfg['num_shots'] = int(num_shots)
    cfg['checkpoint_path'] = checkpoint_path
    cfg['label_delimiter'] = label_delimiter
    cfg['retrieval_method'] = retrieval_method
    cfg['temperature'] = float(temperature)
    cfg['do_sample'] = do_sample
    cfg['seed'] = seed
    main(cfg)
    