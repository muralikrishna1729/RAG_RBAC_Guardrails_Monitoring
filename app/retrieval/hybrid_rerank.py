import math 
from typing import List,Dict, Tuple
from sentence_transformers import CrossEncoder
_reranker_instance = None 

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        try:
             _reranker_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker_instance = None
    return _reranker_instance

def reciprocal_rank_fusion(dense_docs: List, sparse_docs: List, k:int= 60)-> List[Tuple[object, float]]:
    """
    Combines dense vector search & sparse lexical BM25 results using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score(d) = sum( 1 / (k + rank(d)) )
    """
    scores = {}
    doc_map = {}
    for rank, doc in enumerate(dense_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc 
        scores[doc_id] = scores.get(doc_id, 0.0)+(1.0/(k+rank+1))

    
    for rank, doc in enumerate(sparse_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
    
    reranked = sorted(scores.items(), key = lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in reranked]

def rerank_documents(query: str, docs: List, top_k: int = 3) -> List:
    """
    Re-scores retrieved candidate chunks using a Cross-Encoder model.
    """
    reranker = get_reranker()
    if not reranker or not docs:
        return docs[:top_k]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]