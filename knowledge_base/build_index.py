"""
Build the RAG index from a PDF and persist it to disk.

Usage:
    python knowledge_base/build_index.py                   # build from config.pdf
    python knowledge_base/build_index.py /path/to/new.pdf  # swap in a new PDF
"""

import os
import json
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from knowledge_base.pdf_loader import PDFLoader, clean_text
from knowledge_base.chunker import SemanticChunker
from knowledge_base.embeddings import Embeddings
from knowledge_base.vector_store import VectorStore
from knowledge_base.bm25_index import BM25Index


def build(pdf_path=None):
    pdf_path = pdf_path or config.PDF_PATH
    t0 = time.time()

    print(f"=== Building RAG index from: {pdf_path} ===")

    # 1. Extract
    loader = PDFLoader(pdf_path)
    pages = loader.extract_pages()
    print(f"Extracted {len(pages)} pages")

    # 2. Clean + chunk
    pages = [{"page": p["page"], "text": clean_text(p["text"])} for p in pages]
    chunker = SemanticChunker()
    chunks = chunker.chunk_pages(pages)
    print(f"Created {len(chunks)} semantic chunks")

    # Save chunks
    os.makedirs(config.KB_DIR, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to {config.CHUNKS_PATH}")

    # 3. Embeddings
    emb = Embeddings(config.EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = emb.encode(texts)
    print(f"Encoded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # 4. FAISS vector store
    metadata = [{"topic": c["topic"], "page": c["page"], "content": c["content"]} for c in chunks]
    vec_store = VectorStore(config.FAISS_INDEX_PATH, config.VECTOR_META_PATH)
    vec_store.build(embeddings, metadata)

    # 5. BM25
    bm25 = BM25Index(config.BM25_DUMP_PATH)
    bm25.build(metadata)

    # 6. Persist embeddings cache
    emb_build_cache = os.path.join(config.KB_DIR, "embeddings_cache")
    # Save embeddings alongside chunks for faster future startup
    np_save = embeddings
    os.makedirs(emb_build_cache, exist_ok=True)
    # (embeddings are re-derivable; FAISS already holds them)

    elapsed = time.time() - t0
    print(f"=== Index built in {elapsed:.1f}s: {len(chunks)} chunks ===")
    print(f"  Chunks:    {config.CHUNKS_PATH}")
    print(f"  FAISS:     {config.FAISS_INDEX_PATH}")
    print(f"  Metadata:  {config.VECTOR_META_PATH}")
    print(f"  BM25:      {config.BM25_DUMP_PATH}")
    return chunks


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    build(target)
