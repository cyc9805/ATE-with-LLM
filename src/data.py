import json
import copy
import torch
import Stemmer
from copy import deepcopy
from glob import glob
from transformers import AutoTokenizer
from typing import Any, Optional, Dict, List
from prompt_template import *
from vector_store import RetrieverForTextDistance, VectorStoreQueryMode
from llama_index.retrievers.bm25 import BM25Retriever

IGNORE_INDEX = -100

def initialize_gpt():
    import openai
    import os
    from dotenv import load_dotenv
    global model
    load_dotenv(".env")
    model = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

def preprocess_data(x, 
                    i: int, 
                    tokenizer: AutoTokenizer, 
                    model_name: str, 
                    dataset_name: str,
                    retriever: Any, 
                    retrieved_cache_path: str, 
                    num_shots: int, 
                    is_train_dataset: bool, 
                    retrieval_method: Optional[str] = 'default',
                    label_delimiter: str=', ',
                    seed: Optional[int] = 42):   
    
    # global total_retrieved_results   
    no_term_token = '' if 'bart' in model_name else 'No term'

    if 'bart' not in model_name:
        domain_for_task = x['domain'].strip()
        demonstrations_str = ''
        retrieved_results = []
        # No texts were retrieved previously
        if retriever is not None and num_shots > 0:
            if retrieval_method == 'default_w_ins':
                query_text = DECODER_RETRIEVER_QUERY_TEMPLATE.format(domain_for_task, x['text'])
            else:
                query_text = x['text']
            
            for retrieved_result in retriever.retrieve(query_text):                    
                metadata = retrieved_result.metadata
                metadata['score'] = retrieved_result.score
                retrieved_results.append(metadata)
                
        # Texts were retrieved previously
        elif retriever is None and num_shots > 0:
            with open(retrieved_cache_path, 'r') as f:
                retrieved_results = json.load(f)[str(i)][:num_shots] 
            
        x['retrieved_result'] = str(retrieved_results)

        if num_shots > 0:
            for i, retrieved_result in enumerate(retrieved_results, 1):
                text = retrieved_result['text']
                label = retrieved_result['label']
                
                if len(label) == 0:
                    label = [no_term_token]
                    
                domain_name = retrieved_result['domain']
                final_response = label_delimiter.join(label)
                demonstrations_str += DEMONSTRATION_TEMPLATE.format(i, domain_name, text, final_response)+'\n' 
                
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(domain_for_task, demonstrations_str)
            
    if len(x['label'])==0:
        label_str = no_term_token
    else:
        label_str = label_delimiter.join(x['label'])

    user_prompt = USER_PROMPT_TEMPLATE.format(x['text'])
    if 'llama' in model_name:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    elif 'gemma' in model_name or 'mistral' in model_name:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        messages = [{"role": "user", "content": combined_prompt}]
    elif 'bart' in model_name:
        messages = x['text']
        model_inputs = tokenizer(messages, text_target=label_str)
        x['input_ids']  = model_inputs['input_ids']
        x['labels'] = model_inputs['labels']
        del x['label']
        if 'label' in x:
            del x['label']
        return x
    elif 'gpt' in model_name:
        global model
        messages = [{"role":"system", "content":system_prompt}, {"role":"user", "content":user_prompt}]
        response = model.chat.completions.create(
                model=model_name,
                messages=messages,            
            )
        answer = response.choices[0].message.content
        x['input_texts'] = f"{system_prompt}\n\n{user_prompt}"
        x['predictions'] = answer
        x['labels'] = label_str
        del x['label']
        if 'label' in x:
            del x['label']
        return x
    
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    labels = tokenizer(label_str)['input_ids']
        
    x['labels'] = labels
    x['input_ids'] = input_ids

    if 'label' in x:
        del x['label']
    
    return x

def create_retriever(index, dataset_name, num_shots, retrieval_method, seed, rebuild_index=False):
    vector_store_query_mode = [mode for mode in VectorStoreQueryMode if mode.value == retrieval_method][0]
    if vector_store_query_mode == VectorStoreQueryMode.FASTKASSIM:
        if isinstance(index, dict):
            metadata_dict = index
        else:
            metadata_dict = index._vector_store.data.metadata_dict
        text_ids, sentences = [], []
        for key in metadata_dict:
            text_ids.append(key)
            sentences.append(metadata_dict[key]['text'])
        retriever = RetrieverForTextDistance(sentences, num_shots, None, text_ids, metadata_dict, dataset_name, vector_store_query_mode, rebuild_index)

    elif vector_store_query_mode == VectorStoreQueryMode.BM25:
        # Use default BM25Retriever
        retriever = BM25Retriever.from_defaults(
                        index=index,
                        similarity_top_k=5,
                        stemmer=Stemmer.Stemmer("english"),     
                        language="english",
                    )            
        retriever.similarity_top_k = num_shots    
    else:
        num_devices = torch.cuda.device_count()
        index._embed_model._model = index._embed_model._model.to(f'cuda:{num_devices-1}')
        retriever = index.as_retriever()
        retriever._vector_store_query_mode = vector_store_query_mode
        retriever._similarity_top_k = num_shots
        retriever._kwargs = {}

        if retrieval_method == 'random':
            retriever._kwargs.update({"seed": seed, "dataset_name": dataset_name})
    
    return retriever