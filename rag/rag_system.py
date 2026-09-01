"""
RAGSystem: the high-level orchestrator used by the FastAPI app.

Loads the persisted index once at startup, keeps the local LLM resident,
and wires together: hybrid retrieval -> confidence routing -> generation.
"""

import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from knowledge_base.embeddings import Embeddings
from knowledge_base.vector_store import VectorStore
from knowledge_base.bm25_index import BM25Index
from knowledge_base.retriever import HybridRetriever
from knowledge_base.answer_generator import AnswerGenerator
from models.local_llm import LocalLLM


class RAGSystem:
    def __init__(self, load_llm=True):
        self.embeddings = Embeddings(config.EMBEDDING_MODEL)

        self.vector_store = VectorStore(config.FAISS_INDEX_PATH, config.VECTOR_META_PATH)
        if not self.vector_store.load():
            raise RuntimeError(
                "FAISS index not found. Run: python knowledge_base/build_index.py"
            )

        self.bm25 = BM25Index(config.BM25_DUMP_PATH)
        if not self.bm25.load():
            raise RuntimeError(
                "BM25 index not found. Run: python knowledge_base/build_index.py"
            )

        self.retriever = HybridRetriever(
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            bm25=self.bm25,
            vector_top_k=config.TOP_K_VECTOR,
            bm25_top_k=config.TOP_K_BM25,
            final_top_k=config.TOP_K_FINAL,
        )

        self.llm = LocalLLM() if load_llm else None
        self.generator = AnswerGenerator(self.llm) if self.llm else None

    def answer(self, question, mode="fast"):
        """Full pipeline. Returns a structured result dict."""
        t0 = time.time()

        retrieved, breakdown = self.retriever.retrieve(question)
        retrieve_ms = (time.time() - t0) * 1000

        top_score = retrieved[0]["score"] if retrieved else 0.0

        gen = self.generator.generate(question, retrieved, top_score, mode=mode) if self.generator else {
            "answer": retrieved[0]["content"] if retrieved else "This information is not available in the knowledge base.",
            "mode": "extracted",
            "generation_ms": 0.0,
            "ttft_ms": 0.0,
        }

        total_ms = (time.time() - t0) * 1000
        return {
            "question": question,
            "answer": gen["answer"],
            "mode": gen["mode"],
            "confidence": round(top_score, 4),
            "evidence": [{
                "topic": r["topic"],
                "page": r["page"],
                "content": r["content"],
                "score": r["score"],
                "vector_score": r["vector_score"],
                "bm25_score": r["bm25_score"],
            } for r in retrieved],
            "retrieval_ms": round(retrieve_ms, 2),
            "generation_ms": round(gen.get("generation_ms", 0.0), 2),
            "ttft_ms": round(gen.get("ttft_ms", 0.0), 2),
            "total_ms": round(total_ms, 2),
            "breakdown": breakdown,
        }

    def stream(self, question):
        """For streaming: yields (event_type, payload)."""
        from models.local_llm import stream_generate  # noqa
        from fastapi.responses import StreamingResponse  # noqa
        # Not implemented in core; the app layers SSE on top of answer().
        return self.answer(question)

    def health(self):
        return {
            "status": "healthy",
            "chunks": self.vector_store.size,
            "embedding_model": config.EMBEDDING_MODEL,
            "generation_model": config.LOCAL_MODEL,
            "llm_loaded": bool(self.llm and self.llm.is_loaded),
            "high_conf": config.HIGH_CONFIDENCE,
            "low_conf": config.LOW_CONFIDENCE,
        }


if __name__ == "__main__":
    system = RAGSystem(load_llm=False)
    while True:
        q = input("\nAsk (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        t0 = time.time()
        r = system.answer(q)
        print(f"\n[confidence={r['confidence']} mode={r['mode']} "
              f"total={r['total_ms']}ms retrieve={r['retrieval_ms']}ms]")
        print(r["answer"][:800])
