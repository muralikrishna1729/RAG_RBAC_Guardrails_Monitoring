"""
Local, CPU-only sentence-transformers embeddings.

The model remains lazy-loaded. Document embedding is performed in bounded
batches with smaller-batch retries and strict output validation so a partial
or malformed embedding result cannot be mistaken for a complete batch.
"""
import math
import os
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "resources" / "models" / "all-MiniLM-L6-v2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_sentence_transformer(model_path: str = None, device: str = "cpu"):
    """Load the sentence-transformer model (local folder first, hub only once)."""
    from sentence_transformers import SentenceTransformer

    local_dir = Path(model_path) if model_path else DEFAULT_MODEL_DIR
    if local_dir.exists() and any(local_dir.iterdir()):
        return SentenceTransformer(str(local_dir), device=device)

    model = SentenceTransformer(DEFAULT_MODEL_NAME, device=device)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(local_dir))
    return model


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Lazy CPU embeddings with validated, retryable document batches."""

    def __init__(self, model_path: str = None, device: str = "cpu", batch_size: int = 32):
        self._model_path = model_path
        self._device = device
        self._batch_size = max(1, batch_size)
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_path or DEFAULT_MODEL_NAME

    @property
    def sentence_transformer(self):
        if self._model is None:
            self._model = _get_sentence_transformer(self._model_path, self._device)
        return self._model

    def _encode_batch(self, batch: List[str], start_index: int) -> List[List[float]]:
        last_error = None
        for attempt in range(3):
            effective_batch_size = max(1, self._batch_size // (2 ** attempt))
            try:
                encoded = self.sentence_transformer.encode(
                    batch,
                    batch_size=effective_batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                vectors = [vector.tolist() for vector in encoded]
                expected_count = len(batch)
                if len(vectors) != expected_count:
                    raise ValueError(
                        f"Expected {expected_count} vectors, received {len(vectors)}."
                    )

                expected_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
                for offset, vector in enumerate(vectors):
                    if expected_dimension and len(vector) != expected_dimension:
                        raise ValueError(
                            f"Vector {offset} has dimension {len(vector)}; "
                            f"expected {expected_dimension}."
                        )
                    if not all(math.isfinite(float(value)) for value in vector):
                        raise ValueError(f"Vector {offset} contains NaN or infinite values.")
                return vectors
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Embedding batch starting at document {start_index} "
            f"(size={len(batch)}) failed after 3 attempts: {last_error}"
        ) from last_error

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start:start + self._batch_size])
            vectors.extend(self._encode_batch(batch, start))

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Incomplete embedding result: expected {len(texts)}, received {len(vectors)}."
            )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "not loaded (lazy)"
        return (
            f"LocalSentenceTransformerEmbeddings(model={self._model_path or DEFAULT_MODEL_NAME}, "
            f"device={self._device}, status={status})"
        )


_embeddings = None


def get_embeddings() -> LocalSentenceTransformerEmbeddings:
    """Return the shared embeddings instance. The model loads on first use."""
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalSentenceTransformerEmbeddings()
    return _embeddings
