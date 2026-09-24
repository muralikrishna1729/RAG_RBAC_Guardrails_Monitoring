"""
Local, CPU-only sentence-transformers embeddings.

Replaces the `langchain-huggingface` / `HuggingFaceEmbeddings` wrapper with a
direct `sentence_transformers` integration that:

- Runs on CPU only: `device="cpu"` is forced (torch is the lightweight "+cpu"
  build installed from https://download.pytorch.org/whl/cpu - no CUDA/GPU).
- Uses a locally downloaded model: the first run resolves the model once and
  saves a copy to resources/models/all-MiniLM-L6-v2; every later run loads
  purely from that local folder with zero network access.
- Loads lazily: importing this module does NOT load the model. The weights are
  only loaded on the first embed call, not at application start.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings

# Resolve the project root from this file: app/embeddings/local_embeddings.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "resources" / "models" / "all-MiniLM-L6-v2"

# Keep tokenizer parallelism off (CPU-friendly, avoids fork warnings).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_sentence_transformer(model_path: str = None, device: str = "cpu"):
    """Load the sentence-transformer model (local folder first, hub only once)."""
    from sentence_transformers import SentenceTransformer  # imported lazily on purpose

    local_dir = Path(model_path) if model_path else DEFAULT_MODEL_DIR

    if local_dir.exists() and any(local_dir.iterdir()):
        # Purely local load - no network/hub access at all.
        return SentenceTransformer(str(local_dir), device=device)

    # First run: resolve from the local HF cache (downloads only if missing),
    # then persist a local copy so future runs never need the hub again.
    model = SentenceTransformer(DEFAULT_MODEL_NAME, device=device)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(local_dir))
    return model


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    LangChain `Embeddings` implementation backed directly by sentence-transformers.

    The underlying model is created lazily on the first embed call and is
    pinned to CPU, loading from a local model folder when available.
    """

    def __init__(self, model_path: str = None, device: str = "cpu", batch_size: int = 32):
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._model = None  # Nothing loaded yet - lazy.

    @property
    def model(self):
        """Lazy singleton: loads the model on first access, not at startup."""
        if self._model is None:
            self._model = _get_sentence_transformer(self._model_path, self._device)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "not loaded (lazy)"
        return (
            f"LocalSentenceTransformerEmbeddings(model={self._model_path or DEFAULT_MODEL_NAME}, "
            f"device={self._device}, status={status})"
        )


# Process-wide singleton so every module shares one lazily-loaded model.
_embeddings = None


def get_embeddings() -> LocalSentenceTransformerEmbeddings:
    """Return the shared embeddings instance. The model loads on first use."""
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalSentenceTransformerEmbeddings()
    return _embeddings
