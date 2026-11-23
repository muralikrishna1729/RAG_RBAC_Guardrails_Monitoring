import hashlib
import shutil
import time
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.embeddings import get_embeddings
from app.utils.index_state import write_index_generation


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
        source_path = Path(doc.metadata.get("source", "")).resolve()
        if not str(source_path):
            continue

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
                # Fall back to the loaded record when the original file is unavailable.
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
    """Rebuild the current Chroma store and publish a new cache generation marker.

    NOTE: This function is still a destructive rebuild. Safe staging/activation is
    the next ingestion milestone; the generation marker only prevents stale
    application caches from being reused after a successful rebuild.
    """
    embeddings = get_embeddings()
    store_path = Path(persist_directory)

    if store_path.exists():
        for attempt in range(3):
            try:
                for item in store_path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                break
            except (PermissionError, OSError):
                if attempt == 2:
                    raise RuntimeError(
                        "Could not clear the existing Chroma store: another process "
                        "(e.g. a running Streamlit app / API server) is holding it. "
                        "Stop that app, then re-run the ingestion."
                    )
                time.sleep(1)
    else:
        store_path.mkdir(parents=True, exist_ok=True)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    generation = write_index_generation(persist_directory)
    try:
        from app.retrieval.hybrid_rerank import reset_bm25_cache
        reset_bm25_cache()
    except Exception:
        # Generation-scoped keys still invalidate stale BM25 entries in other processes.
        pass

    print(f"Success: Stored {len(chunks)} chunks in ChromaDB at {persist_directory}")
    print(f"Published index generation: {generation}")
    return vector_db


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
