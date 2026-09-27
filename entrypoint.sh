#!/bin/bash
set -e

# Perform initial document ingestion into ChromaDB if vector store is missing or empty
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo "⚡ ChromaDB directory missing or empty. Running document ingestion script..."
    python -m app.ingestion.ingest
    echo "✅ Document ingestion completed successfully."
else
    echo "ℹ️ Existing ChromaDB vector store found."
fi

# Execute passed command (default: Streamlit UI or passed container command)
exec "$@"
