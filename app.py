"""
FastAPI Server for the ETL Interview RAG System.
Local MLX generation only. Grounded answers from the PDF knowledge base.
"""

import json
import os
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from config import HOST, PORT
from models.local_llm import scrub_answer
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

# Speech to text. Constructed once; reports unavailable rather than failing if no
# key is configured, so the on-device path still works.
from models.transcription import Transcriber
transcriber = Transcriber()


class QueryRequest(BaseModel):
    question: str
    mode: str = "fast"
    # Which generation model to answer with. Unknown or omitted falls back to
    # the default, so older clients keep working.
    model: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    # Which answer style routing chose: grounded, scenario, or open.
    # Undeclared fields are stripped by FastAPI, so this must be listed.
    style: str = ""
    # Which generation model actually answered.
    model: str = ""
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


@app.get("/api/voice/config")
def voice_config():
    """What the client should use for speech input."""
    return {
        "engines": [
            {"id": "browser", "label": "Instant (on-device)",
             "blurb": "Words appear as you speak. Weaker on terms like SCD."},
            {"id": "hybrid", "label": "Instant + corrected",
             "blurb": "Live text while speaking, then corrected for accuracy."},
            {"id": "whisper", "label": "Accurate (online)",
             "blurb": "About 0.7s after you stop. Best on ETL terms."},
        ],
        "default_engine": (config.VOICE_DEFAULT_ENGINE
                           if transcriber.is_loaded else "browser"),
        "default_language": config.VOICE_DEFAULT_LANGUAGE,
        "languages": [
            {"id": "en-IN", "label": "English (India)"},
            {"id": "en-US", "label": "English (US)"},
            {"id": "en-GB", "label": "English (UK)"},
            {"id": "hi-IN", "label": "Hindi"},
            {"id": "kn-IN", "label": "Kannada"},
            {"id": "te-IN", "label": "Telugu"},
            {"id": "ta-IN", "label": "Tamil"},
            {"id": "mr-IN", "label": "Marathi"},
        ],
        "silence_ms": config.VOICE_SILENCE_MS,
        # Only offer the online engines when a key is actually configured.
        "online_available": transcriber.is_loaded,
    }


@app.post("/api/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Transcribe a recorded clip and normalise it to the notes' vocabulary."""
    if not transcriber.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Online transcription is not configured. Use on-device instead.",
        )
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(data) > config.VOICE_MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Recording too long.")

    # Whisper picks the language itself when none is given, which is better than
    # forcing the wrong one.
    lang = (language or "").split("-")[0] or None
    result = transcriber.transcribe(
        data, filename=audio.filename or "audio.webm", language=lang
    )
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])
    return {
        "text": result["text"],
        "raw": result["raw"],
        "transcribe_ms": round(result["ms"]),
    }


@app.post("/api/voice/correct")
def voice_correct(payload: dict):
    """Normalise text the browser already transcribed, with no audio upload.

    The browser engine is fast but mangles domain terms; this maps them back to
    the notes' spelling so retrieval can match them."""
    from models.transcription import correct_transcript
    text = (payload or {}).get("text", "")
    return {"text": correct_transcript(text)}


@app.get("/api/models")
def list_models():
    """Selectable generation models, with their measured tradeoffs."""
    return {
        "models": qa_system.models(),
        "default": qa_system.registry.default_id if qa_system.registry else None,
    }


@app.get("/api/sample-questions")
def get_sample_questions():
    return [
        {"text": "What is SCD Type 2?", "category": "SCD"},
        {"text": "What happens when customer address changes?", "category": "SCD"},
        {"text": "Difference between TRUNCATE and DELETE", "category": "SQL"},
        {"text": "What are the types of joins?", "category": "SQL"},
        {"text": "How to find department-wise 2nd max salary", "category": "SQL Queries"},
        {"text": "What is a Surrogate Key and why is it used?", "category": "Data Warehouse"},
        {"text": "Star Schema vs Fact and Dimension tables", "category": "Data Warehouse"},
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
    mode = req.mode if req.mode in ("fast", "large") else "fast"
    return qa_system.answer(q, mode=mode, model_id=req.model)


@app.post("/api/ask-stream")
def ask_question_stream(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    mode = req.mode if req.mode in ("fast", "large") else "fast"

    t0 = time.time()

    # 1. Retrieve, then choose an answer style (cheap: no generation).
    retrieved, breakdown = qa_system.retriever.retrieve(q)
    retrieve_ms = (time.time() - t0) * 1000
    routed = qa_system.generator.route(q, retrieved)

    # 2. Resolve the requested model. First use of a model loads it here, which
    #    is why the client shows a "warming up" hint on first switch.
    model_id, llm = qa_system.pick_model(req.model)

    # No model available: return the best section verbatim in one packet.
    if not (llm and llm.is_loaded):
        payload = {
            "type": "complete",
            "question": q,
            "answer": retrieved[0]["content"] if retrieved else "",
            "mode": "extracted",
            "style": routed["style"],
            "model": model_id or "",
            "confidence": routed["confidence"],
            "evidence": qa_system.evidence(retrieved),
            "retrieval_ms": round(retrieve_ms, 2),
            "generation_ms": 0.0,
            "ttft_ms": 0.0,
            "breakdown": breakdown,
            "total_ms": round((time.time() - t0) * 1000, 2),
        }

        async def single():
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        return StreamingResponse(single(), media_type="text/event-stream")

    # 3. Stream the LLM tokens live.
    participants = llm.generate(
        q, retrieved, stream=True, style=routed["style"], mode=mode
    )

    def stream_tokens():
        meta = {
            "type": "meta",
            "question": q,
            "mode": "generated",
            "style": routed["style"],
            "model": model_id or "",
            "confidence": routed["confidence"],
            "evidence": qa_system.evidence(retrieved),
            "retrieval_ms": round(retrieve_ms, 2),
            "breakdown": breakdown,
        }
        yield f"data: {json.dumps(meta)}\n\n".encode("utf-8")

        full = []
        first_ttft_ms = None
        for tok, ttft in participants:
            if tok == "":
                break
            if first_ttft_ms is None:
                first_ttft_ms = ttft
            full.append(tok)
            yield f"data: {json.dumps({'type': 'token', 'text': tok})}\n\n".encode("utf-8")

        # The first line is streamed unscrubbed to keep time-to-first-token
        # low, so the assembled answer is scrubbed here before it is saved to
        # the student's history.
        done = {
            "type": "done",
            "answer": scrub_answer("".join(full)),
            "ttft_ms": round(first_ttft_ms or 0, 2),
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
    # The app OBJECT is passed, not the "app:app" import string. With the
    # string, uvicorn re-imports this module, which runs the module-level
    # RAGSystem(...) a second time and loads the LLM and embedding model twice
    # (doubling startup time and peak memory).
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
