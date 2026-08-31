"""
FastAPI Server for the ETL Interview RAG System.
Local MLX generation only. Grounded answers from the PDF knowledge base.
"""

import json
import os
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import HOST, PORT
from rag.rag_system import RAGSystem

app = FastAPI(title="ETL Interview RAG Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG system at startup (index + local LLM resident in memory)
qa_system = RAGSystem(load_llm=True)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    confidence: float
    evidence: list
    retrieval_ms: float
    generation_ms: float
    ttft_ms: float
    total_ms: float
    breakdown: dict


@app.get("/api/health")
def health_check():
    h = qa_system.health()
    return h


@app.get("/api/sample-questions")
def get_sample_questions():
    return [
        {"text": "What is SCD Type 2?", "category": "SCD"},
        {"text": "What happens when customer address changes?", "category": "SCD"},
        {"text": "Difference between TRUNCATE and DELETE", "category": "SQL"},
        {"text": "How to find 2nd highest salary in SQL", "category": "SQL Queries"},
        {"text": "What is a Surrogate Key and why is it used?", "category": "Data Warehouse"},
        {"text": "Star Schema vs Snowflake Schema", "category": "Data Warehouse"},
        {"text": "Explain the Defect Life Cycle", "category": "Defect Life Cycle"},
        {"text": "What are the Levels of Testing?", "category": "Testing"},
    ]


@app.get("/api/sources/{page}")
def get_source(page: int):
    """Return the raw text of a given PDF page for source cross-referencing."""
    from knowledge_base.pdf_loader import PDFLoader
    from config import PDF_PATH
    loader = PDFLoader(PDF_PATH)
    pages = loader.extract_pages()
    for p in pages:
        if p["page"] == page:
            return {"page": page, "text": p["text"]}
    raise HTTPException(status_code=404, detail=f"Page {page} not found")


@app.post("/api/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return qa_system.answer(q)


@app.post("/api/ask-stream")
def ask_question_stream(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t0 = time.time()

    # 1. Retrieve + route (fast, no generation yet)
    retrieved, breakdown = qa_system.retriever.retrieve(q)
    top_score = retrieved[0]["score"] if retrieved else 0.0
    retrieve_ms = (time.time() - t0) * 1000

    gen_result = qa_system.generator.generate(q, retrieved, top_score)

    # Route through the quick path (unsupported/extracted/LLM-fallbacks) that
    # does NOT stream — send as a single packet.
    if gen_result["mode"] != "generated" or not (
        qa_system.llm and qa_system.llm.is_loaded
    ):
        payload = {
            "type": "complete",
            "question": q,
            "answer": gen_result["answer"],
            "mode": gen_result["mode"],
            "confidence": round(top_score, 4),
            "evidence": retrieved,
            "retrieval_ms": round(retrieve_ms, 2),
            "generation_ms": round(gen_result.get("generation_ms", 0), 2),
            "ttft_ms": round(gen_result.get("ttft_ms", 0), 2),
            "breakdown": breakdown,
            "total_ms": round((time.time() - t0) * 1000, 2),
        }
        async def single():
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        return StreamingResponse(single(), media_type="text/event-stream")

    # 2. Otherwise stream the LLM tokens over SSE
    def stream_tokens():
        meta = {
            "type": "meta",
            "question": q,
            "mode": "generated",
            "confidence": round(top_score, 4),
            "evidence": retrieved,
            "retrieval_ms": round(retrieve_ms, 2),
            "breakdown": breakdown,
            "ttft_ms": round((time.time() - t0) * 1000, 2),
        }
        yield f"data: {json.dumps(meta)}\n\n".encode("utf-8")

        gen = qa_system.llm.generate(q, retrieved, stream=True)
        full = []
        first_ttft = None
        for tok, ttft in gen:
            if tok == "":
                break
            if first_ttft is None and ttft:
                first_ttft = ttft
            full.append(tok)
            yield f"data: {json.dumps({'type': 'token', 'text': tok})}\n\n".encode("utf-8")

        done = {
            "type": "done",
            "answer": "".join(full),
            "ttft_ms": round(first_ttft or 0, 2),
            "generation_ms": round((time.time() - t0) * 1000, 2),
            "total_ms": round((time.time() - t0) * 1000, 2),
        }
        yield f"data: {json.dumps(done)}\n\n".encode("utf-8")

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
