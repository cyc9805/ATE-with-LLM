import numpy as np
import pickle
import heapq
import nltk
import logging
import os
import copy
import fkassim.FastKassim as fkassim
import multiprocessing as mp
from tqdm import tqdm
from utils import CustomFastKassim 
from multiprocessing import Pool
from nltk import word_tokenize
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional, cast
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_embeddings_learner
)
from llama_index.core.utils import concat_dirs
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.simple import SimpleVectorStore, _build_metadata_filter_fn
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.base.embeddings.base import similarity, SimilarityMode
from llama_index.core.postprocessor import LLMRerank

nltk.download('wordnet')
nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

FastKassim = CustomFastKassim(fkassim.FastKassim.LTK)

TREE_PARSED_DIR = '/data_x/yongchan/ATE/tree_parsed_{0}.pkl'
NAMESPACE_SEP = "__"
DEFAULT_VECTOR_STORE = "default"

class VectorStoreQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    DEFAULT_W_INS = "default_w_ins"

    # Synatax similarity    
    FASTKASSIM = "fastkassim"

    # BM25
    BM25 = 'bm25'
    
    # Random
    RANDOM = "random"

@dataclass
class NodeWithOnlyMetadataAndScore:
    metadata: Dict[str, Any]
    score: Optional[float]

class CustomVectorStore(SimpleVectorStore):
    domain_name_to_embedding: dict = {}
    
    def calculate_embedding_similarity(
        self):
        metadata_dict = self.data.to_dict()['metadata_dict']
        embedding_dict = self.data.embedding_dict
        remove_node_ids = []
        for node_id, metadata in metadata_dict.items():
            if metadata['label'] == metadata['category'] == 'SPECIAL SENTENCE' and metadata['domain'] not in self.domain_name_to_embedding:
                self.domain_name_to_embedding[metadata['domain']] = self.data.embedding_dict[node_id]
                remove_node_ids.append(node_id)
                print('Special sentence:', metadata['text'], 'domain:', metadata['domain'])
        
        for node_id in remove_node_ids:
            del embedding_dict[node_id]
            del metadata_dict[node_id]
            
        new_embedding_dict = {}
        for node_id, embedding in tqdm(embedding_dict.items()):
            domain = metadata_dict[node_id]['domain']
            special_embedding = self.domain_name_to_embedding[domain]
            cosine_similarity = similarity(
                    special_embedding,
                    embedding
                )
            new_embedding_dict[node_id] = cosine_similarity
        
        self.data.metadata_dict = metadata_dict
        self.data.embedding_dict = new_embedding_dict
        
    def query(
            self,
            query: VectorStoreQuery,
            **kwargs: Any,
        ) -> VectorStoreQueryResult:
            """Get nodes for response."""
            # Prevent metadata filtering on stores that were persisted without metadata.
            if (
                query.filters is not None
                and self.data.embedding_dict
                and not self.data.metadata_dict
            ):
                raise ValueError(
                    "Cannot filter stores that were persisted without metadata. "
                    "Please rebuild the store with metadata to enable filtering."
                )
            # Prefilter nodes based on the query filter and node ID restrictions.
            query_filter_fn = _build_metadata_filter_fn(
                lambda node_id: self.data.metadata_dict[node_id], query.filters
            )

            if query.node_ids is not None:
                available_ids = set(query.node_ids)
                def node_filter_fn(node_id: str) -> bool:
                    return node_id in available_ids

            else:
                def node_filter_fn(node_id: str) -> bool:
                    return True

            node_ids = []
            embeddings = []
            # TODO: consolidate with get_query_text_embedding_similarities
            for node_id, embedding in self.data.embedding_dict.items():
                if node_filter_fn(node_id) and query_filter_fn(node_id):
                    node_ids.append(node_id)
                    embeddings.append(embedding)

            query_embedding = cast(List[float], query.query_embedding)

            if query.mode == VectorStoreQueryMode.RANDOM:      
                seed = kwargs.get('seed', 42)
                
                np.random.seed(seed)
                selected_ids = np.random.choice(range(len(node_ids)), query.similarity_top_k, replace=False)
                
                node_ids = [node_ids[i] for i in selected_ids]
                embeddings = [embeddings[i] for i in selected_ids]
                
                top_similarities = [None] * query.similarity_top_k
                return VectorStoreQueryResult(similarities=top_similarities, ids=node_ids)
                
                
            elif query.mode in [VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.DEFAULT_W_INS]:
                if query.mode in [VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.DEFAULT_W_INS]:
                    similarity_fn = similarity
                top_similarities, top_ids = get_top_k_embeddings(
                    query_embedding,
                    embeddings,
                    similarity_fn,
                    similarity_top_k=query.similarity_top_k,
                    embedding_ids=node_ids,
                )

            else:
                raise ValueError(f"Invalid query mode: {query.mode}")

            return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
        
class RetrieverForTextDistance:
    def __init__(
            self, 
            sentences, 
            similarity_top_k=None, 
            similarity_cutoff=None, 
            text_ids=None, 
            metadata_dict=None,
            dataset_name=None,
            query_mode=VectorStoreQueryMode.FASTKASSIM,
            reparse_document=False):
        
        self.sentences = sentences
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        self.text_ids = text_ids
        self.metadata_dict = metadata_dict
        self.query_mode = query_mode
        self.dataset_name = dataset_name
        cpu_cores = mp.cpu_count()
        if query_mode == VectorStoreQueryMode.FASTKASSIM:
            _tree_parsed_dir = TREE_PARSED_DIR.format(dataset_name)
            if not os.path.exists(_tree_parsed_dir) or reparse_document:
                num_processes = min(cpu_cores//2, len(sentences))  
                with Pool(num_processes) as pool:
                    _sentences = pool.map(FastKassim.parse_document_with_index, [[i,s] for i,s in enumerate(sentences)])
                    _sentences = sorted(_sentences, key=lambda x: x[0])
                    _sentences = [s[1] for s in _sentences]
                    if not reparse_document:
                        with open(_tree_parsed_dir, 'wb') as f:
                            pickle.dump(_sentences, f)
            else:
                with open(_tree_parsed_dir, 'rb') as f:
                    _sentences = pickle.load(f)
    
            self.sentences = _sentences

    def retrieve(self, text):
        simliarities, top_ids = get_top_k_sentences(
            text,
            self.sentences,
            self.dataset_name,
            similarity_top_k=self.similarity_top_k,
            similarity_cutoff=self.similarity_cutoff,
            embedding_or_text_ids=self.text_ids,
            query_mode=self.query_mode
        )
        query_result = VectorStoreQueryResult(similarities=simliarities, ids=top_ids)

        node_with_scores = []
        for ind, idx in enumerate(query_result.ids):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
                metadata = self.metadata_dict[idx]
            node_with_scores.append(NodeWithOnlyMetadataAndScore(metadata=metadata, score=score))
        return node_with_scores
    

def get_top_k_sentences(
    query_str: str,
    sentences: List[Any],
    dataset_name: str=None, 
    similarity_top_k: Optional[int] = None,
    similarity_cutoff: Optional[float] = None,
    embedding_or_text_ids: Optional[List] = None,   
    query_mode: VectorStoreQueryMode = VectorStoreQueryMode.FASTKASSIM,
) -> Tuple[List[float], List]:
    
    if embedding_or_text_ids is None:
        embedding_or_text_ids = list(range(len(embedding_or_text_ids)))

    elif query_mode == 'fastkassim':
        distance_metric = calculate_fastkassim
        query_str = FastKassim.parse_document(query_str)
        
    similarity_heap: List[Tuple[float, Any]] = []
    for i, sentence in enumerate(sentences):
        similarity = distance_metric(copy.deepcopy(query_str), sentence)
        if similarity_cutoff is None or similarity > similarity_cutoff:
            heapq.heappush(similarity_heap, (similarity, embedding_or_text_ids[i]))
            if similarity_top_k and len(similarity_heap) > similarity_top_k:
                heapq.heappop(similarity_heap)
    result_tups = sorted(similarity_heap, key=lambda x: x[0], reverse=True)

    result_similarities = [s for s, _ in result_tups]
    result_texts = [n for _, n in result_tups]

    return result_similarities, result_texts

def calculate_fastkassim(candidate, reference):
    return FastKassim.compute_similarity_preparsed(reference, candidate)