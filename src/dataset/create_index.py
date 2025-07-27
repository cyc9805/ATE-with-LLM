import os
import torch
from tqdm import tqdm
from llama_index.core import Settings
from datasets import load_from_disk
from llama_index.core import (SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
VectorStoreIndex,
Document)
from transformers import AutoTokenizer

embed_model = "local:BAAI/bge-large-en-v1.5"        # Options: "local:BAAI/bge-large-en-v1.5", "local:BAAI/bge-en-icl"
dataset_dirs = [
    'ACTER/huggingface',
    'ACL-RD/huggingface',
    'BCGM/huggingface'
    ]

for dataset_dir in tqdm(dataset_dirs):
    ds = load_from_disk(dataset_dir)

    documents = []
    for i, entry in enumerate(ds['train']):
        input_text = entry['text']
        label = entry['label']
        domain = entry['domain']
        if 'acter' in dataset_dir.lower():
            category = entry['unique_label']
        elif 'acl-rd' in dataset_dir.lower():
            category = entry['category']
        elif 'bcgm' in dataset_dir.lower():
            category = entry['category']

        if embed_model == "local:BAAI/bge-en-icl":
            text_for_embedding = f"Domain: {domain}\nSentence: {input_text}"
        else:
            text_for_embedding = input_text
            
        document = Document(text=text_for_embedding)
        document.metadata = {'text': input_text, 'label': label, 'category': category, 'domain': domain}

        if i < 10:
            print('input_text:', text_for_embedding)
            print('label:', label)
            print('category:', category)
            print('domain:', domain)
            print()
        documents.append(document)
    
    if 'BCGM' in dataset_dir:
        Settings.chunk_size = max(len(doc.text) for doc in documents)
    else:
        Settings.chunk_size = max(len(doc.text) for doc in documents) + 1200

    Settings.chunk_overlap = 50

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)    
    index_name = os.path.basename(embed_model)
    persist_dir = os.path.join(os.path.dirname(os.path.dirname(dataset_dir)),'index', index_name)
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    del index
    torch.cuda.empty_cache()