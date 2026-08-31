"""
Local embeddings using BAAI/bge-small-en-v1.5.
Generates embeddings once and persists them to disk for fast startup.
Runs entirely on Apple Silicon via sentence-transformers/mps.
"""

import os
import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer


class Embeddings:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", cache_dir="knowledge_base/embeddings_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self._cache_path = os.path.join(cache_dir, "embeddings.npy")
        self._meta_path = os.path.join(cache_dir, "meta.json")
        os.makedirs(cache_dir, exist_ok=True)

    def load_model(self):
        if self.model is None:
            t0 = time.time()
            print(f"Loading embedding model '{self.model_name}'...")
            # Use MPS (Metal) on Apple Silicon for speed
            self.model = SentenceTransformer(self.model_name)
            print(f"Embedding model loaded in {(time.time()-t0)*1000:.0f}ms")
        return self.model

    def encode(self, texts, batch_size=32):
        """Encode a list of texts to normalized numpy embeddings."""
        model = self.load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def encode_query(self, text):
        """Encode a single query (normalized)."""
        embed = self.encode([text])[0]
        # bge model recommends adding this prefix for retrieval queries
        return embed

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------
    def save(self, embeddings, source_ids):
        """Persist embeddings and their mapping to disk."""
        np.save(self._cache_path, embeddings)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({"model": self.model_name, "source_ids": source_ids}, f)
        print(f"Persisted {len(source_ids)} embeddings to {self._cache_path}")

    def load(self):
        """Load persisted embeddings. Returns (embeddings, source_ids) or None."""
        if not (os.path.exists(self._cache_path) and os.path.exists(self._meta_path)):
            return None
        embeddings = np.load(self._cache_path)
        with open(self._meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model") != self.model_name:
            return None
        return embeddings, meta["source_ids"]
