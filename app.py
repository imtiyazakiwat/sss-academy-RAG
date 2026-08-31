"""
FastAPI Server for ETL Interview Q&A System with Rotated Gemini Proxy
"""

import time
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import uvicorn

from qa_pipeline import (
    QASystem,
    EXACT_MATCH_THRESHOLD,
    NEAR_MATCH_THRESHOLD,
    TOP_K_CONTEXT,
    GROQ_MODEL,
    EMBEDDING_MODEL,
    DEFAULT_GROQ_KEY
)

app = FastAPI(title="ETL Interview Q&A Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_system = QASystem(api_key=DEFAULT_GROQ_KEY)

class QueryRequest(BaseModel):
    question: str
    api_key: Optional[str] = None
    engine: Optional[str] = "mlx"  # "mlx" or "groq"

class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    engine: str = "mlx"
    similarity: float
    latency_ms: float
    generation_ms: float = 0.0
    topic: str = ""
    top_matches: list = []

@app.get("/api/health")
def health_check():
    mlx_active = bool(qa_system.mlx_generator and qa_system.mlx_generator.is_loaded)
    return {
        "status": "healthy",
        "indexed_count": len(qa_system.index.qa_data),
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GROQ_MODEL,
        "mlx_active": mlx_active,
        "mlx_model": "Google Gemma 2 2B + DoRA (Apple M4)",
        "default_engine": "mlx" if mlx_active else "groq",
        "has_groq_key": True,
        "exact_threshold": EXACT_MATCH_THRESHOLD,
        "near_threshold": NEAR_MATCH_THRESHOLD,
    }

@app.get("/api/sample-questions")
def get_sample_questions():
    return [
        {"text": "Tell me about yourself / Self Introduction", "category": "General"},
        {"text": "Explain your project architecture & data flow stages", "category": "Architecture"},
        {"text": "difference between truncate and delete", "category": "SQL"},
        {"text": "what is scd type 2 and how do you test it", "category": "SCD"},
        {"text": "how to find 2nd highest salary in SQL", "category": "SQL Queries"},
        {"text": "What is the difference between Star Schema and Snowflake Schema?", "category": "Data Warehouse"},
        {"text": "What do you do if a developer pushes back and says a bug is expected behavior?", "category": "Situational (Trained MLX)"},
        {"text": "How do you test real-time Kafka streaming data into Snowflake?", "category": "Streaming / DWH"}
    ]

@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    key_to_use = req.api_key.strip() if (req.api_key and req.api_key.strip()) else DEFAULT_GROQ_KEY
    chosen_engine = req.engine if req.engine in ["mlx", "groq"] else "mlx"
    
    res = qa_system.answer(q, api_key=key_to_use, engine=chosen_engine)
    return QueryResponse(
        question=q,
        answer=res["answer"],
        mode=res["mode"],
        engine=res.get("engine", chosen_engine),
        similarity=res["similarity"],
        latency_ms=res["latency_ms"],
        generation_ms=res.get("generation_ms", 0.0),
        topic=res.get("topic", "General"),
        top_matches=res.get("top_matches", [])
    )

@app.post("/api/ask-stream")
async def ask_question_stream(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    key_to_use = req.api_key.strip() if (req.api_key and req.api_key.strip()) else DEFAULT_GROQ_KEY
    chosen_engine = req.engine if req.engine in ["mlx", "groq"] else "mlx"
    t0 = time.time()

    # 1. Search index
    results = qa_system.index.search(q, top_k=TOP_K_CONTEXT)
    top = results[0]

    # 2. If exact match, send immediately in single SSE packet
    if top["similarity"] >= EXACT_MATCH_THRESHOLD:
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        payload = {
            "type": "exact",
            "mode": "exact_match",
            "engine": chosen_engine,
            "similarity": round(top["similarity"], 3),
            "topic": top["topic"],
            "matched_question": top["question"],
            "text": top["answer"],
            "latency_ms": elapsed_ms,
            "top_matches": results
        }
        async def exact_stream():
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            
        return StreamingResponse(exact_stream(), media_type="text/event-stream")

    # 3. Stream unknown response chunk by chunk (MLX or Groq)
    async def stream_tokens():
        meta = {
            "type": "meta",
            "mode": "generated_unknown",
            "engine": chosen_engine,
            "similarity": round(top["similarity"], 3),
            "topic": top.get("topic", "ETL Testing"),
            "matched_question": top["question"],
            "top_matches": results,
            "ttft_ms": round((time.time() - t0) * 1000, 1)
        }
        yield f"data: {json.dumps(meta)}\n\n".encode("utf-8")

        full_text = []
        if chosen_engine == "mlx" and qa_system.mlx_generator and qa_system.mlx_generator.is_loaded:
            for token in qa_system.mlx_generator.stream_generate(q):
                full_text.append(token)
                chunk_data = {"type": "token", "text": token}
                yield f"data: {json.dumps(chunk_data)}\n\n".encode("utf-8")
        else:
            qa_system.generator.set_api_key(key_to_use)
            for token in qa_system.generator.stream_generate(q, results):
                full_text.append(token)
                chunk_data = {"type": "token", "text": token}
                yield f"data: {json.dumps(chunk_data)}\n\n".encode("utf-8")

        total_ms = round((time.time() - t0) * 1000, 1)
        done_data = {
            "type": "done",
            "total_latency_ms": total_ms
        }
        yield f"data: {json.dumps(done_data)}\n\n".encode("utf-8")

        # Cache completed answer
        final_answer = "".join(full_text).strip()
        if not final_answer.startswith("⚠️"):
            cache_key = f"{chosen_engine}:{q.lower()}"
            qa_system.response_cache[cache_key] = {
                "answer": final_answer,
                "mode": "generated_unknown",
                "engine": chosen_engine,
                "similarity": round(top["similarity"], 3),
                "matched_question": top["question"],
                "topic": top.get("topic", "ETL Testing"),
                "latency_ms": total_ms,
                "generation_ms": total_ms,
                "top_matches": results
            }

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
