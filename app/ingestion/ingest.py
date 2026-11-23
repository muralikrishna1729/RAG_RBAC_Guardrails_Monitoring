import hashlib
import os
import time
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.embeddings import get_embeddings
from app.utils.index_state import write_index_generation


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_documents(data_path: str):
    """Load records with stable document IDs and content-derived version identifiers."""
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

    file_hashes = {}
    record_indexes = {}
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

        record_index = record_indexes.get(source_key, 0)
        record_indexes[source_key] = record_index + 1
        content_hash = file_hashes[source_key]
        document_version_id = hashlib.sha256(
            f"{document_id}:{content_hash}".encode("utf-8")
        ).hexdigest()

        doc.metadata["role"] = role
        doc.metadata["source"] = relative_source
        doc.metadata["document_id"] = document_id
        doc.metadata["content_hash"] = content_hash
        doc.metadata["document_version_id"] = document_version_id
        doc.metadata["record_index"] = record_index
        doc.metadata["record_hash"] = _sha256_bytes(doc.page_content.encode("utf-8"))

    print(f"Loaded {len(all_docs)} document records.")
    return all_docs


def split_documents(all_docs: list):
    """Split records into chunks and assign deterministic chunk identifiers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    for chunk in chunks:
        chunk_hash = _sha256_bytes(chunk.page_content.encode("utf-8"))
        document_version_id = chunk.metadata.get("document_version_id", "unknown")
        record_index = chunk.metadata.get("record_index", 0)
        chunk.metadata["chunk_hash"] = chunk_hash
        chunk.metadata["chunk_id"] = hashlib.sha256(
            f"{document_version_id}:{record_index}:{chunk_hash}".encode("utf-8")
        ).hexdigest()
    print(f"Success: Split into {len(chunks)} chunks.")
    return chunks


def store_in_chroma(chunks: list, persist_directory: str):
    """Stage a new Chroma collection, validate it, then atomically activate it.

    The active collection is never deleted or modified during ingestion. Old
    collections are retained for rollback until a separate retention cleanup runs.
    """
    if not chunks:
        raise ValueError("Refusing to activate an empty document index.")

    store_path = Path(persist_directory)
    store_path.mkdir(parents=True, exist_ok=True)

    lock_path = store_path / ".ingestion.lock"
    lock_fd = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, f"pid={os.getpid()} started={time.time()}".encode("utf-8"))
    except FileExistsError as exc:
        raise RuntimeError(
            "Another ingestion job holds the Chroma writer lock. If the previous "
            "process crashed, verify it is stopped before removing the stale lock: "
            f"{lock_path}"
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
        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        vector_db.add_documents(chunks, ids=chunk_ids)

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
    print(f"Total document records loaded: {len(raw_docs)}")

    if raw_docs:
        doc_chunks = split_documents(raw_docs)
        store_in_chroma(doc_chunks, db_path)
        print("\n--- INGESTION COMPLETE ---")
        print(f"Sample Chunk Metadata: {doc_chunks[0].metadata}")
