"""
RAGSystem: the high-level orchestrator used by the FastAPI app.

Loads the persisted index once at startup, keeps the local LLM resident,
and wires together: hybrid retrieval -> confidence routing -> generation.
"""

import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from knowledge_base.embeddings import Embeddings
from knowledge_base.vector_store import VectorStore
from knowledge_base.bm25_index import BM25Index
from knowledge_base.retriever import HybridRetriever, Lexicon
from knowledge_base.answer_generator import AnswerGenerator
from models.model_registry import ModelRegistry


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

        # Parent sections for small-to-big expansion. Children are what get
        # matched; parents are what the LLM reads.
        if not os.path.exists(config.PARENTS_PATH):
            raise RuntimeError(
                "Parent sections not found. Run: python knowledge_base/build_index.py"
            )
        with open(config.PARENTS_PATH, "r", encoding="utf-8") as f:
            self.parents = json.load(f)
        print(f"Loaded {len(self.parents)} parent sections")

        # Term statistics over the notes. Built before the retriever because
        # ranking uses them to prefer sections that hold the question's rare,
        # decisive words, and routing uses them to spot uncovered topics.
        self.lexicon = Lexicon([p["content"] for p in self.parents])

        self.retriever = HybridRetriever(
            embeddings=self.embeddings,
            vector_store=self.vector_store,
            bm25=self.bm25,
            parents=self.parents,
            lexicon=self.lexicon,
            vector_top_k=config.TOP_K_VECTOR,
            bm25_top_k=config.TOP_K_BM25,
            final_top_k=config.TOP_K_FINAL,
        )

        # Selectable generation models. The default is loaded now; the others
        # load on first use so startup stays quick.
        self.registry = ModelRegistry() if load_llm else None
        self.generator = AnswerGenerator(lexicon=self.lexicon)
        if self.registry:
            if config.PRELOAD_ALL_MODELS:
                self.registry.preload()
            else:
                self.registry.get(self.registry.default_id)

        # Force the lazy retrieval models to load and run one throwaway pass, so
        # the first real question does not pay the ~9 s embedding-model load.
        warm_ms = self.retriever.warmup()
        print(f"Warmup complete in {warm_ms:.0f} ms")

    @property
    def llm(self):
        """The default model. Kept so existing scripts and benchmark.py, which
        assume a single resident model, continue to work."""
        if not self.registry:
            return None
        return self.registry.get(self.registry.default_id)[1]

    def pick_model(self, model_id=None):
        """Resolve a requested model id to (id, llm), loading it if needed."""
        if not self.registry:
            return None, None
        return self.registry.get(model_id)

    @staticmethod
    def evidence(retrieved):
        """Citable evidence for the UI. `content` here is the matched fragment,
        not the whole parent section, so the panel stays readable."""
        return [{
            "topic": r.get("topic", ""),
            "heading": r.get("heading", ""),
            "page": r.get("page"),
            "page_label": r.get("page_label", ""),
            "content": r.get("child_content") or r.get("content", ""),
            "score": r.get("score"),
            "vector_score": r.get("vector_score"),
            "bm25_score": r.get("bm25_score"),
            "lexical": r.get("lexical"),
        } for r in retrieved]

    def answer(self, question, mode="fast", model_id=None):
        """Full pipeline. Returns a structured result dict."""
        t0 = time.time()

        retrieved, breakdown = self.retriever.retrieve(question)
        retrieve_ms = (time.time() - t0) * 1000

        resolved_id, llm = self.pick_model(model_id)
        if llm:
            gen = self.generator.generate(question, retrieved, mode=mode, llm=llm)
        else:
            gen = {
                "answer": retrieved[0]["content"] if retrieved else "",
                "mode": "extracted",
                "style": "none",
                "confidence": 0.0,
                "generation_ms": 0.0,
                "ttft_ms": 0.0,
            }

        total_ms = (time.time() - t0) * 1000
        return {
            "question": question,
            "answer": gen["answer"],
            "mode": gen["mode"],
            "style": gen.get("style", ""),
            "model": resolved_id or "",
            "confidence": gen.get("confidence", 0.0),
            "evidence": self.evidence(retrieved),
            "retrieval_ms": round(retrieve_ms, 2),
            "generation_ms": round(gen.get("generation_ms", 0.0), 2),
            "ttft_ms": round(gen.get("ttft_ms", 0.0), 2),
            "total_ms": round(total_ms, 2),
            "breakdown": breakdown,
        }

    def models(self):
        return self.registry.describe() if self.registry else []

    def health(self):
        return {
            "status": "healthy",
            "chunks": self.vector_store.size,
            "parent_sections": len(self.parents),
            "embedding_model": config.EMBEDDING_MODEL,
            "models": self.models(),
            "default_model": self.registry.default_id if self.registry else None,
            "draft_model": config.DRAFT_MODEL,
            "grounded_lexical": config.GROUNDED_LEXICAL,
            "open_lexical": config.OPEN_LEXICAL,
        }


if __name__ == "__main__":
    system = RAGSystem(load_llm=False)
    while True:
        q = input("\nAsk (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        r = system.answer(q)
        print(f"\n[confidence={r['confidence']} style={r['style']} "
              f"mode={r['mode']} total={r['total_ms']}ms "
              f"retrieve={r['retrieval_ms']}ms]")
        print(r["answer"][:1200])
