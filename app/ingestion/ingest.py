import os
import time
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.embeddings import get_embeddings
from app.utils.index_state import write_index_generation

import hashlib


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_documents(data_path: str):
    """Load Markdown/CSV records with stable document IDs and source content hashes."""
    root = Path(data_path).resolve()
    loaders = [
        DirectoryLoader(
            data_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        ),
        DirectoryLoader(data_path, glob="**/*.csv", loader_cls=CSVLoader),
    ]

    all_docs = []
    for loader in loaders:
        all_docs.extend(loader.load())

    seen_records = set()
    unique_docs = []
    file_hashes = {}

    for doc in all_docs:
        source_value = doc.metadata.get("source")
        if not source_value:
            continue
        source_path = Path(source_value).resolve()
        role = source_path.parent.name.lower()
        try:
            relative_source = source_path.relative_to(root).as_posix()
        except ValueError:
            relative_source = source_path.name

        document_id = hashlib.sha256(
            f"{role}:{relative_source}".encode("utf-8")
        ).hexdigest()

        source_key = str(source_path)
        if source_key not in file_hashes:
            try:
                file_hashes[source_key] = _sha256_bytes(source_path.read_bytes())
            except OSError:
                file_hashes[source_key] = _sha256_bytes(doc.page_content.encode("utf-8"))

        record_hash = _sha256_bytes(doc.page_content.encode("utf-8"))
        dedup_key = (document_id, record_hash)
        if dedup_key in seen_records:
            continue
        seen_records.add(dedup_key)

        doc.metadata["role"] = role
        doc.metadata["document_id"] = document_id
        doc.metadata["content_hash"] = file_hashes[source_key]
        doc.metadata["record_hash"] = record_hash
        doc.metadata["source"] = relative_source
        unique_docs.append(doc)

    print(f"Loaded {len(all_docs)} records; retained {len(unique_docs)} unique records.")
    return unique_docs


def split_documents(all_docs: list):
    """Split source records into overlapping chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    for chunk in chunks:
        chunk.metadata["chunk_hash"] = _sha256_bytes(chunk.page_content.encode("utf-8"))
    print(f"Success: Split into {len(chunks)} chunks.")
    return chunks


def store_in_chroma(chunks: list, persist_directory: str):
    """Build a new collection, validate it, then atomically activate it.

    The current active collection is never deleted or modified during staging.
    A failed embedding/write leaves the previous collection active.
    """
    if not chunks:
        raise ValueError("Refusing to activate an empty document index.")

    store_path = Path(persist_directory)
    store_path.mkdir(parents=True, exist_ok=True)

    # One ingestion writer at a time across processes sharing this persistence path.
    lock_path = store_path / ".ingestion.lock"
    lock_fd = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, f"pid={os.getpid()} started={time.time()}".encode("utf-8"))
    except FileExistsError as exc:
        raise RuntimeError(
            "Another ingestion job holds the Chroma writer lock. "
            "If the previous process crashed, verify it is stopped before removing "
            f"the stale lock file: {lock_path}"
        ) from exc

    generation = uuid4().hex
    collection_name = f"rag_{generation[:16]}"
    vector_db = None
    activated = False

    try:
        vector_db = Chroma(
            collection_name=collection_name,
            persist_directory=str(store_path),
            embedding_function=get_embeddings(),
        )
        vector_db.add_documents(chunks)

        stored_count = vector_db._collection.count()
        if stored_count != len(chunks):
            raise RuntimeError(
                f"Staged index validation failed: expected {len(chunks)} chunks, "
                f"found {stored_count}."
            )

        published_generation = write_index_generation(
            str(store_path),
            collection_name=collection_name,
            generation=generation,
        )
        activated = True

        try:
            from app.retrieval.hybrid_rerank import reset_bm25_cache
            reset_bm25_cache()
        except Exception:
            # Other processes use generation-scoped BM25 keys and self-invalidate.
            pass

        print(
            f"Activated Chroma collection {collection_name} with "
            f"{stored_count} chunks (generation={published_generation})."
        )
        return vector_db
    except Exception:
        if vector_db is not None and not activated:
            try:
                vector_db.delete_collection()
            except Exception:
                pass
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    data_path = "./resources/data"
    db_path = "./chroma_db"
    print(f"Starting load from: {data_path}...")

    raw_docs = load_documents(data_path)
    print(f"Total documents loaded: {len(raw_docs)}")

    if raw_docs:
        doc_chunks = split_documents(raw_docs)
        vector_store = store_in_chroma(doc_chunks, db_path)
        print("\n--- INGESTION COMPLETE ---")
        print(f"Sample Chunk Metadata: {doc_chunks[0].metadata}")
