import re
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_reranker_instance = None
_bm25_cache: Dict[str, dict] = {}

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


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens with light suffix stripping (openings->open, days->day)."""
    tokens = re.findall(r"\w+", text.lower())
    stemmed = []
    for token in tokens:
        if len(token) > 4:
            for suffix in ("ings", "ing", "ies", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                    token = token[: -len(suffix)]
                    break
        stemmed.append(token)
    return stemmed


def reset_bm25_cache(role: str = None) -> None:
    """Drops cached BM25 indexes (call after re-ingesting documents)."""
    if role is None:
        _bm25_cache.clear()
    else:
        _bm25_cache.pop(role, None)


def _load_role_documents(role: str, vectorstore=None) -> List[Document]:
    """Loads every chunk the role is permitted to see (Chroma doubles as the doc store)."""
    try:
        if vectorstore is None:
            from app.embeddings import get_embeddings
            from langchain_chroma import Chroma
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=get_embeddings())
        role_filter = None if role == "admin" else {"$or": [{"role": role}, {"role": "general"}]}
        stored = vectorstore.get(where=role_filter) if role_filter else vectorstore.get()
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        return [
            Document(page_content=doc, metadata=metadatas[i] if i < len(metadatas) else {})
            for i, doc in enumerate(documents)
        ]
    except Exception:
        return []


def get_bm25_index(role: str, vectorstore=None):
    """Builds and caches a BM25 index over the role-permitted chunks (per role, per process)."""
    if role in _bm25_cache:
        return _bm25_cache[role]

    docs = _load_role_documents(role, vectorstore)
    if not docs:
        return None

    from rank_bm25 import BM25Okapi  # lazy import; hybrid degrades to dense-only without it
    index = {"bm25": BM25Okapi([_tokenize(d.page_content) for d in docs]), "docs": docs}
    _bm25_cache[role] = index
    return index


def bm25_search(query: str, role: str, top_k: int = 6, vectorstore=None) -> List:
    """Sparse BM25 retrieval over the role-permitted chunks (empty list on any failure)."""
    index = get_bm25_index(role, vectorstore)
    if not index:
        return []
    try:
        scores = index["bm25"].get_scores(_tokenize(query))
    except Exception:
        return []
    ranked = sorted(zip(index["docs"], scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k] if score > 0]