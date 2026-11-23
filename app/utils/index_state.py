"""Small local index-generation marker shared by retrieval and cache layers.

The marker changes only after a new Chroma index has been written successfully.
It lets each application process detect index replacement without Redis or a
distributed cache invalidation service.
"""
import os
from pathlib import Path
from uuid import uuid4


GENERATION_FILENAME = ".index_generation"


def get_index_generation(persist_directory: str = "./chroma_db") -> str:
    marker = Path(persist_directory) / GENERATION_FILENAME
    try:
        value = marker.read_text(encoding="utf-8").strip()
        return value or "unversioned"
    except OSError:
        return "unversioned"


def write_index_generation(persist_directory: str = "./chroma_db") -> str:
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    generation = uuid4().hex
    marker = directory / GENERATION_FILENAME
    temporary = directory / f"{GENERATION_FILENAME}.tmp"
    temporary.write_text(generation, encoding="utf-8")
    os.replace(temporary, marker)
    return generation
