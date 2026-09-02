"""
Central configuration for the ETL Interview RAG System.
All paths, thresholds, and model settings live here.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=None):
    """Read .env into the environment without adding a dependency.

    Values already set in the environment win, so a rotated key can be exported
    for one run without editing the file. Never logged.
    """
    path = path or os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()


def groq_api_key():
    """Read at call time so the key can be rotated without a code change."""
    return (os.environ.get("GROQ_API_KEY") or "").strip()

# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join(BASE_DIR, "SSS_CADEMY_NOTES New.pdf")
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")

# Derived index artifacts (persisted to disk, built once by build_index.py)
CHUNKS_PATH = os.path.join(KB_DIR, "extracted_chunks.json")
# Parent sections for small-to-big retrieval: children are indexed and matched,
# then expanded to their parent section before generation.
PARENTS_PATH = os.path.join(KB_DIR, "parent_sections.json")
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
# ---------------------------------------------------------------------------
# Selectable generation models
# ---------------------------------------------------------------------------
# Students pick a model per question. Both are loaded lazily and then kept
# resident, so switching is instant after first use. Numbers below are measured
# on an M4 with benchmark.py (40 questions, greedy decoding) and are shown in
# the UI so the speed/completeness tradeoff is an informed choice.
MODELS = {
    "fast-3b": {
        "repo": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "label": "Quick (3B)",
        "blurb": "Answers start in about 1.4s. Shorter answers.",
        "accuracy": "82%",
        "ttft_ms": 1404,
        "answer_words": 36,
        "memory_gb": 2.5,
        "default": True,
    },
    "accurate-4b": {
        "repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510",
        "label": "Detailed (4B)",
        "blurb": "Answers start in about 2.7s. Fuller answers that list every point.",
        "accuracy": "88%",
        "ttft_ms": 2694,
        "answer_words": 92,
        "default": False,
    },
    # Runs on Groq, so it needs internet and a GROQ_API_KEY, and the question
    # leaves this machine. Much larger model than anything that fits locally.
    "best-online": {
        "provider": "groq",
        "repo": "qwen/qwen3.8-27b",
        "label": "Best (online)",
        "blurb": "A 27B model on Groq. Needs internet; questions leave this device.",
        "accuracy": "-",
        "ttft_ms": None,
        "answer_words": None,
        "default": False,
    },
}

# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
GROQ_TIMEOUT_S = 45

# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------
# whisper-large-v3-turbo measured at 640-860 ms round trip for 3-4 s of speech,
# and it gets "SCD Type 2", "de-normalized" and "BCNF" right. whisper-large-v3
# is slightly slower with no accuracy gain on this vocabulary.
TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
# Which engine the UI uses by default. "browser" streams interim words with no
# perceptible delay; "whisper" is accurate but ~700 ms; "hybrid" shows the
# browser's live text and corrects it with Whisper on stop.
VOICE_DEFAULT_ENGINE = "hybrid"
VOICE_DEFAULT_LANGUAGE = "en-IN"
# Stop listening after this much silence.
VOICE_SILENCE_MS = 1500
# Reject absurdly large uploads outright.
VOICE_MAX_UPLOAD_BYTES = 8_000_000
# Reasoning models spend part of the budget thinking before answering, and
# gpt-oss-120b was observed consuming an entire 700-token budget on reasoning
# and returning empty content. Remote budgets are scaled up to leave room.
GROQ_TOKEN_HEADROOM = 2.0
# Alternatives verified working on the free tier: openai/gpt-oss-120b (set
# reasoning_effort="low" via the model entry), openai/gpt-oss-20b (fastest,
# ~970 tok/s), qwen/qwen3.6-27b (emits reasoning blocks, stripped automatically).

DEFAULT_MODEL_ID = next(
    (k for k, v in MODELS.items() if v.get("default")), next(iter(MODELS))
)

# Backwards compatibility: single-model entry point used by benchmark.py
# (--model overrides it) and by any script that builds a LocalLLM directly.
LOCAL_MODEL = MODELS[DEFAULT_MODEL_ID]["repo"]

# Load every model at startup instead of on first use. Off by default so boot
# stays quick; the second model then costs a one-time ~2s load when first chosen.
PRELOAD_ALL_MODELS = False
MAX_TOKENS = 256
MAX_TOKENS_LARGE = 512          # large mode: extended budget for detailed notes
# Comparison tables need more room than a bullet list: a 14-row table ran past
# the 256-token cap and was cut mid-row, which renders as broken markdown.
MAX_TOKENS_TABLE = 480
TEMPERATURE = 0.2

# Mild repetition penalty. Greedy/low-temperature decoding cannot escape a loop,
# and the 3B was observed repeating one comparison-table row until the token
# limit. Kept low so repeating a term across table rows stays possible.
REPETITION_PENALTY = 1.08
REPETITION_CONTEXT = 40

# Context budget fed to the LLM. Prefill (first-token) time scales with this.
#
# Retrieval now returns whole PARENT SECTIONS rather than 200-char fragments,
# so one context block is larger but complete: asking "what are joins" needs
# all seven join types, which is ~2400 chars in this PDF. The budget is raised
# to fit one full section, and the TTFT cost is bought back by making the
# static instruction block a reusable prompt-cache prefix (see local_llm.py).
MAX_CONTEXT_CHARS = 2600

# Adaptive context, measured rather than assumed.
#
# Time-to-first-token is 94% model prefill and 6% retrieval, at roughly
# 2.8 ms per prefilled token on this hardware. Cutting context therefore does
# buy latency, but it buys it directly out of accuracy, measured on the
# 40-question suite with the 3B:
#
#   budget 2600, 1 section   88% correct, 12/40 under 1s   <- chosen
#   budget 1600, 1 section   85% correct,  7/40 under 1s
#   budget  900, 1 section   88% correct,  6/40 under 1s
#   budget  500, 1 section   75% correct, 27/40 under 1s
#   budget 2600, 2 sections  85% correct,  5/40 under 1s
#
# Sending 2 sections lost on both axes: the notes restate topics, so the second
# slot usually held a near-duplicate while still costing full prefill. Focused
# trimming gained one question and halved the sub-1s count. Both mechanisms are
# kept and exposed to benchmark.py (--topk-context, --focus-chars) so the
# tradeoff stays measurable, but the defaults are the configuration that
# actually won.
TOP_K_CONTEXT = 1
CONTEXT_FOCUS_CHARS = 10 ** 9   # effectively off; see the table above
CONTEXT_PREAMBLE_CHARS = 320    # section opening kept with a focused window

# Two retrieved sections are treated as the same material above this word
# overlap. These notes restate topics (OLTP vs OLAP on both page 5 and page 54,
# PK vs Surrogate Key on both 51 and 57), and measured top-2 Jaccard is 0.99+
# for such queries, so without this the second slot is spent on a duplicate.
CONTEXT_DEDUPE_JACCARD = 0.55

# KV-cache reuse. Entries hold the system prefix per answer style plus recent
# full prompts, so a follow-up question that retrieves the same section skips
# re-prefilling it. A 3B entry runs tens of MB, hence the byte cap.
# Measured: 12 entries held 569 MB for the 3B. The 4B's are roughly twice that,
# and both models can be resident, so the cap is set to leave headroom on 16 GB.
PROMPT_CACHE_ENTRIES = 8
PROMPT_CACHE_MAX_BYTES = 900_000_000

# ---------------------------------------------------------------------------
# Retrieval Settings
# ---------------------------------------------------------------------------
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
# Parent sections, not fragments, so fewer are needed.
TOP_K_FINAL = 3

# ---------------------------------------------------------------------------
# Answer-style routing
# ---------------------------------------------------------------------------
# The system never refuses a question. These thresholds only choose HOW to
# answer: strictly from the notes, by reasoning over them, or from the model's
# own expertise while keeping the notes' vocabulary and voice.
#
# Scales, measured on this corpus:
#   lexical  0.0-1.0  fraction of meaningful question words found in the section
#   vector   0.0-1.0  cosine similarity from bge-small
#   rerank   unbounded logit from ms-marco-MiniLM; relevant hits observed 4-8,
#            irrelevant below 0 (calibrate() maps this to 0-1 for display)
GROUNDED_LEXICAL = 0.60   # strong keyword overlap -> answer strictly from notes
GROUNDED_VECTOR = 0.78    # or strong semantic match alone
GROUNDED_RERANK = 4.00    # or a confident cross-encoder verdict
# Coverage is now idf-weighted, so its scale is lower than the old plain token
# fraction: genuinely covered questions measure as low as 0.27 ("daily activity
# in your company", where only "daily" and "activity" carry weight). Detecting
# uncovered topics is UNKNOWN_TOPIC_SHARE's job now, so this floor only catches
# retrieval that missed entirely.
OPEN_LEXICAL = 0.12

# A question is treated as outside the notes when this share of its INFORMATION
# (idf-weighted, stem- and synonym-aware) sits in words the notes never use.
# Presence alone was too blunt: "Star vs Snowflake" contains one absent word of
# four yet is fully covered, while "read Python scripts" contains one absent
# word that is the entire question. Measured on the 22 interview questions,
# 0.30 separates them once stemming and synonym aliasing are applied first.
UNKNOWN_TOPIC_SHARE = 0.30

# ---------------------------------------------------------------------------
# Optional speculative-decoding draft model.
# A small model drafts tokens that the main model verifies, raising decode
# throughput. It does not reduce time-to-first-token. Leave None to disable.
# ---------------------------------------------------------------------------
DRAFT_MODEL = None

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8000
