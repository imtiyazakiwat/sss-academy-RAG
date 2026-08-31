"""
BM25 keyword index using rank_bm25.
Catches exact interview terms that semantic search may miss
(SCD Type 2, Fact Table, Surrogate Key, Truncate, etc.)
"""

import os
import json
import pickle
import re
import time


def tokenize(text: str):
    """Simple whitespace + punctuation tokenizer, lowercased."""
    text = text.lower()
    # Keep alphanumeric tokens and common technical tokens like 'scd_type_2'
    tokens = re.findall(r"[a-z0-9_]+", text)
    return tokens


class BM25Index:
    def __init__(self, dump_path="knowledge_base/bm25_dump.pkl"):
        self.dump_path = dump_path
        self.bm25 = None
        self.doc_ids = []   # parallel to indexed docs, points into metadata

    def build(self, metadata):
        """metadata: list of {topic, page, content}"""
        from rank_bm25 import BM25Okapi
        corpus = [tokenize(m["content"]) for m in metadata]
        self.bm25 = BM25Okapi(corpus)
        self.doc_ids = list(range(len(metadata)))
        self._save()
        return self

    def _save(self):
        os.makedirs(os.path.dirname(self.dump_path), exist_ok=True)
        with open(self.dump_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "doc_ids": self.doc_ids}, f)
        print(f"Saved BM25 index ({len(self.doc_ids)} docs) to {self.dump_path}")

    def load(self):
        if not os.path.exists(self.dump_path):
            return False
        with open(self.dump_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        print(f"Loaded BM25 index ({len(self.doc_ids)} docs) from {self.dump_path}")
        return True

    def search(self, query, top_k=10):
        if self.bm25 is None:
            self.load()
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        t0 = time.time()
        scores = self.bm25.get_scores(q_tokens)
        latency_ms = (time.time() - t0) * 1000
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top:
            if scores[i] <= 0:
                continue
            results.append({
                "doc_id": self.doc_ids[i],
                "score": float(scores[i]),
            })
        return results, latency_ms

    @property
    def size(self):
        return len(self.doc_ids) if self.doc_ids else 0
