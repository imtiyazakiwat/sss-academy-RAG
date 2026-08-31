"""
Central configuration for the ETL Interview RAG System.
All paths, thresholds, and model settings live here.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join(BASE_DIR, "SSS_CADEMY_NOTES New.pdf")
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

# Derived index artifacts (persisted to disk, built once by build_index.py)
CHUNKS_PATH = os.path.join(KB_DIR, "extracted_chunks.json")
FAISS_INDEX_PATH = os.path.join(KB_DIR, "faiss_index.bin")
VECTOR_META_PATH = os.path.join(KB_DIR, "vector_metadata.json")
BM25_DUMP_PATH = os.path.join(KB_DIR, "bm25_dump.pkl")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Local Generation Model (RAW base model, NO LoRA adapter)
# ---------------------------------------------------------------------------
LOCAL_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
# LOCAL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"  # 16GB+ RAM
MAX_TOKENS = 512
TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Retrieval Settings
# ---------------------------------------------------------------------------
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5

# Confidence routing thresholds (hybrid score scale: 0.5 weak ... 1.1 strong)
HIGH_CONFIDENCE = 1.05  # >= 1.05 -> near-verbatim match, return chunk directly
LOW_CONFIDENCE = 0.75    # <  0.75 -> "not available in knowledge base"

# Lexical relevance gate: a match counts as supported only if the raw
# vector similarity is strong, OR there is substantial keyword overlap.
# (Genuine on-topic chunks here all have vector >= ~0.71; off-topic false
#  positives sit lower, so this cleanly rejects them without an LLM call.)
MIN_VECTOR_FOR_WEAK_LEXICAL = 0.65
MIN_LEXICAL_FOR_WEAK_VECTOR = 0.60

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000
