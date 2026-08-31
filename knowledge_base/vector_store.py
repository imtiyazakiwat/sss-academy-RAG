"""
FAISS vector store.
Stores chunk embeddings for fast similarity search.
Persisted to disk and loaded at startup (never rebuilt during inference).
"""

import os
import json
import numpy as np
import time


class VectorStore:
    def __init__(self, index_path="knowledge_base/faiss_index.bin",
                 meta_path="knowledge_base/vector_metadata.json"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata = []   # list of {topic, page, content} aligned with vectors

    def build(self, embeddings, metadata):
        """Build a FAISS index from normalized embeddings."""
        import faiss
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vectors
        self.index.add(embeddings)
        self.metadata = metadata
        self._save()
        return self

    def _save(self):
        import faiss
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"Saved FAISS index ({self.index.ntotal} vectors) to {self.index_path}")

    def load(self):
        import faiss
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            return False
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"Loaded FAISS index ({self.index.ntotal} vectors) from {self.index_path}")
        return True

    def search(self, query_emb, top_k=10):
        """Return top_k (index, score) results."""
        if self.index is None:
            self.load()
        t0 = time.time()
        # Normalize query vector
        query_emb = query_emb / np.linalg.norm(query_emb)
        scores, indices = self.index.search(query_emb[np.newaxis, :], top_k)
        latency_ms = (time.time() - t0) * 1000
        results = []
        for i, score in zip(indices[0], scores[0]):
            i = int(i)
            if i < 0:
                continue
            results.append({
                "index": i,
                "score": float(score),
                **self.metadata[i],
            })
        return results, latency_ms

    @property
    def size(self):
        return self.index.ntotal if self.index is not None else 0
