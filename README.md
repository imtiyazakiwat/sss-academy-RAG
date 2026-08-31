# SSS Academy RAG — ETL Interview Assistant (Retrieval-First RAG)

A true RAG system for ETL Testing interview questions, grounded **strictly** in the
`SSS_CADEMY_NOTES New.pdf` classroom notes. Runs 100% locally on Apple Silicon (M4)
with MLX. No fine-tuning, no Kaggle, no cloud API.

## Why RAG instead of fine-tuning?

| | Old (LoRA fine-tune) | New (RAG) |
|---|---|---|
| Knowledge source | Locked in model weights | Retrieved PDF chunks |
| Update notes | Retrain on Kaggle | Replace PDF + re-index |
| Terminology fidelity | Drifts from synthetic data | Exact from PDF |
| Maintenance | Complex pipeline | `python knowledge_base/build_index.py` |
| Out-of-scope handling | Hallucinates | "Not available in knowledge base" |

For an 81-page PDF, fine-tuning is unnecessary — the notes are injected directly
into the LLM context at inference time, giving zero drift and instant updates.

## Architecture

```
User Question
      │
      ▼
┌──────────────────────────────────────────────┐
│           Hybrid Retriever                   │
│   FAISS (vector) + BM25 (keyword) + rerank   │
│   + query expansion (SCD, Fact Table, …)     │
└──────────────────────────────────────────────┘
      │  Top 5 grounded chunks
      ▼
┌──────────────────────────────────────────────┐
│         Qwen 2.5 3B (RAW, MLX)              │
│   resident in memory, grounded generation   │
└──────────────────────────────────────────────┘
      │
      ▼
        Grounded interview answer with sources
```

**Confidence routing:**
- Retrieval is off-topic (low confidence / no keyword overlap) → returns
  *"This information is not available in the knowledge base."*
- Otherwise the LLM synthesizes a grounded answer, reusing the PDF's terminology.
  The raw top chunk is returned directly only if the local LLM is unavailable.

## Project Structure

```
project/
├── config.py                       # all paths, thresholds, model settings
├── app.py                          # FastAPI server (REST + SSE streaming)
├── knowledge_base/                 # index build + retrieval (source of truth)
│   ├── pdf_loader.py               # PyMuPDF text extraction
│   ├── chunker.py                  # semantic, topic-aligned chunking
│   ├── embeddings.py               # BAAI/bge-small-en-v1.5 (local MPS)
│   ├── vector_store.py             # FAISS index (persisted)
│   ├── bm25_index.py               # rank_bm25 keyword index (persisted)
│   ├── retriever.py                # hybrid retrieval + query expansion
│   ├── answer_generator.py         # confidence routing + grounded generation
│   └── build_index.py              # REBUILD THE INDEX (run on PDF change)
├── models/
│   └── local_llm.py                # raw Qwen 2.5 3B via MLX (no LoRA)
├── rag/
│   └── rag_system.py               # high-level orchestrator
├── static/                         # web UI (history, sources, confidence)
└── SSS_CADEMY_NOTES New.pdf        # THE knowledge base (replace to update)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install pymupdf faiss-cpu rank-bm25 sentence-transformers mlx mlx-lm fastapi uvicorn

# Build the index from the PDF (only needed once, or after swapping the PDF)
python knowledge_base/build_index.py            # or: python knowledge_base/build_index.py /path/to/new.pdf

# Start the server
python app.py                                   # http://127.0.0.1:8000
```

The index (chunks, FAISS, BM25) is built once and persisted. Startup loads it
from disk and keeps the model resident — no cold re-indexing during inference.

## Config (`config.py`)

- `EMBEDDING_MODEL` — `BAAI/bge-small-en-v1.5` (fast, local, MPS)
- `LOCAL_MODEL` — `mlx-community/Qwen2.5-3B-Instruct-4bit` (raw; 7B if ≥16GB RAM)
- `TOP_K_VECTOR` / `TOP_K_BM25` / `TOP_K_FINAL` — retrieval candidates
- `LOW_CONFIDENCE` / `MIN_LEXICAL` / `MIN_VECTOR_FOR_WEAK_LEXICAL` — off-topic gate

## API

- `GET /api/health` — status, chunk count, loaded model
- `GET /api/sample-questions` — quick prompts
- `GET /api/sources/{page}` — raw page text for source cross-reference
- `POST /api/ask {"question": "…"}` — full grounded answer (JSON)
- `POST /api/ask-stream {"question": "…"}` — SSE streaming (meta → tokens → done)

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens when the customer address changes?"}'
```

Returns a grounded SCD Type 2 answer with `evidence[]` (topic, page, content),
a confidence score, and retrieval/generation latency breakdowns.

## Updating the Knowledge Base

Drop a new PDF in place of `SSS_CADEMY_NOTES New.pdf` and re-run:

```bash
python knowledge_base/build_index.py path/to/new.pdf
```

No retraining. The system is ready immediately.

## Latency (Apple M4)

- Retrieval (vector + BM25 + rerank): ~10–60ms
- First generation token: ~0.8s
- Full grounded answer: ~2–4s

## License

Educational project for SSS Academy ETL Testing training (HCL Technologies Menards context).
