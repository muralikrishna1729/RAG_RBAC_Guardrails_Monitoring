"""Atomic active-index pointer for the single-company local Chroma deployment.

The active pointer is a tiny JSON file in the Chroma persistence directory.
New indexes are written into a separate collection and become visible only
when this pointer is atomically replaced.
"""
import json
import os
from pathlib import Path
from uuid import uuid4

GENERATION_FILENAME = ".index_generation"
DEFAULT_COLLECTION_NAME = "langchain"


def _read_index_state(persist_directory: str = "./chroma_db") -> dict:
    marker = Path(persist_directory) / GENERATION_FILENAME
    try:
        raw = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return {"generation": "legacy", "collection_name": DEFAULT_COLLECTION_NAME}

    if not raw:
        return {"generation": "legacy", "collection_name": DEFAULT_COLLECTION_NAME}

    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return {
                "generation": str(value.get("generation") or "legacy"),
                "collection_name": str(value.get("collection_name") or DEFAULT_COLLECTION_NAME),
            }
    except json.JSONDecodeError:
        # Backward compatibility with the earlier plain-text generation marker.
        pass

    return {"generation": raw, "collection_name": DEFAULT_COLLECTION_NAME}


def get_index_generation(persist_directory: str = "./chroma_db") -> str:
    return _read_index_state(persist_directory)["generation"]


def get_active_collection_name(persist_directory: str = "./chroma_db") -> str:
    return _read_index_state(persist_directory)["collection_name"]


def write_index_generation(
    persist_directory: str = "./chroma_db",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    generation: str = None,
) -> str:
    """Atomically publish the collection that retrieval should use."""
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    generation = generation or uuid4().hex
    state = {
        "generation": generation,
        "collection_name": collection_name,
    }
    marker = directory / GENERATION_FILENAME
    temporary = directory / f"{GENERATION_FILENAME}.tmp"
    temporary.write_text(json.dumps(state), encoding="utf-8")
    os.replace(temporary, marker)
    return generation
