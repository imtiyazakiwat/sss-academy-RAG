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
MAX_TOKENS = 200
TEMPERATURE = 0.2

# Context budget fed to the LLM. Prefill (first-token) time scales with this,
# so keep it small for a sub-second TTFT on local MPS. We feed only the top
# chunk(s), truncated, which is enough for grounded short answers.
#   ~1400 chars  -> ~0.9s TTFT
#   ~1900 chars  -> ~1.3s TTFT
#   ~3100 chars  -> ~3.2s TTFT
MAX_CONTEXT_CHARS = 1500
TOP_K_CONTEXT = 2

# ---------------------------------------------------------------------------
# Retrieval Settings
# ---------------------------------------------------------------------------
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5

# Confidence routing thresholds (hybrid score scale: 0.5 weak ... 1.1 strong)
HIGH_CONFIDENCE = 1.05  # >= 1.05 -> near-verbatim match, return chunk directly
LOW_CONFIDENCE = 0.75    # <  0.75 -> "not available in knowledge base"

# Lexical relevance gate: a request is "supported" only if the top chunk has
# a STRONG keyword overlap (lexical >= MIN_LEXICAL_FOR_WEAK_VECTOR), OR a very
# strong pure-vector similarity (vector >= MIN_VECTOR_FOR_WEAK_LEXICAL).
#
# Calibration on this PDF knowledge base:
#   - Genuine on-topic matches: vector >= ~0.80 OR lexical >= ~0.60
#   - Borderline false positives ("Power BI/Tableau", "indexing",
#     "Data Lake vs Warehouse"): vector only ~0.65-0.78 with weak lexical,
#     and the LLM tends to hallucinate outside-knowledge answers for these.
# So the vector path is raised so such off-scope questions are rejected fast
# (no LLM call) instead of producing an ungrounded answer.
MIN_VECTOR_FOR_WEAK_LEXICAL = 0.78
MIN_LEXICAL_FOR_WEAK_VECTOR = 0.60

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000
