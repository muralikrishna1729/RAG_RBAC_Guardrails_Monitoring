"""Local, CPU-only, lazily-loaded sentence-transformers embeddings for the app."""

from app.embeddings.local_embeddings import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_NAME,
    LocalSentenceTransformerEmbeddings,
    get_embeddings,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_NAME",
    "LocalSentenceTransformerEmbeddings",
    "get_embeddings",
]
