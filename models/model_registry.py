"""
Registry of selectable generation models.

Students choose between a quick model and a more detailed one per question. The
two differ in a way that is a real product tradeoff rather than a bug (measured
with benchmark.py on 40 questions, greedy decoding):

    Quick (3B)     82% correct, first token ~1.4s, ~36-word answers, 2.5 GB
    Detailed (4B)  88% correct, first token ~2.7s, ~92-word answers, 3.2 GB

Models load lazily on first selection and then stay resident, so only the first
use of a model pays the load cost. Both together fit comfortably in 16 GB.
"""

import threading
import time

import config
from models.local_llm import LocalLLM


class ModelRegistry:
    def __init__(self, specs=None, default_id=None):
        self.specs = dict(specs or config.MODELS)
        self.default_id = default_id or config.DEFAULT_MODEL_ID

        # Honour an explicit single-model override (benchmark.py --model, or a
        # script setting config.LOCAL_MODEL). Without this the registry would
        # silently serve its own default and the override would look like it
        # worked while measuring the wrong model.
        override = getattr(config, "LOCAL_MODEL", None)
        if override and override != self.specs.get(self.default_id, {}).get("repo"):
            match = next(
                (mid for mid, s in self.specs.items() if s["repo"] == override), None
            )
            if match:
                self.default_id = match
            else:
                self.specs["override"] = {
                    "repo": override,
                    "label": override.split("/")[-1],
                    "blurb": "Set explicitly via config.LOCAL_MODEL.",
                }
                self.default_id = "override"
        self._loaded = {}
        # Loading is not thread safe and FastAPI serves requests from a pool, so
        # a first-use race could otherwise load the same weights twice.
        self._lock = threading.Lock()
        self._load_ms = {}

    # -- ids -------------------------------------------------------------
    def resolve(self, model_id):
        """Map a requested id to a known one, falling back to the default."""
        if model_id in self.specs:
            return model_id
        # Accept a raw repo name too, so the API stays usable from scripts.
        for mid, spec in self.specs.items():
            if spec["repo"] == model_id:
                return mid
        return self.default_id

    def get(self, model_id=None):
        """Return a ready LocalLLM for this id, loading it if needed."""
        mid = self.resolve(model_id)
        llm = self._loaded.get(mid)
        if llm is not None:
            return mid, llm

        with self._lock:
            llm = self._loaded.get(mid)      # re-check inside the lock
            if llm is None:
                spec = self.specs[mid]
                t0 = time.time()
                print(f"Loading model '{mid}' ({spec['repo']})...")
                if spec.get("provider") == "groq":
                    from models.remote_llm import RemoteLLM
                    llm = RemoteLLM(
                        model_path=spec["repo"],
                        reasoning_effort=spec.get("reasoning_effort"),
                    )
                else:
                    llm = LocalLLM(model_path=spec["repo"])
                if llm.is_loaded:
                    llm.warmup()             # prefill the cacheable prefixes
                self._loaded[mid] = llm
                self._load_ms[mid] = (time.time() - t0) * 1000
                print(f"Model '{mid}' ready in {self._load_ms[mid]:.0f} ms")
        return mid, self._loaded[mid]

    def preload(self, model_ids=None):
        for mid in (model_ids or list(self.specs)):
            self.get(mid)

    def unload(self, model_id):
        with self._lock:
            llm = self._loaded.pop(self.resolve(model_id), None)
        if llm:
            llm.unload()

    # -- reporting -------------------------------------------------------
    def is_loaded(self, model_id):
        mid = self.resolve(model_id)
        llm = self._loaded.get(mid)
        return bool(llm and llm.is_loaded)

    def describe(self):
        """Public model list for the UI."""
        out = []
        for mid, spec in self.specs.items():
            provider = spec.get("provider", "local")
            llm = self._loaded.get(mid)
            out.append({
                "id": mid,
                "label": spec.get("label", mid),
                "repo": spec["repo"],
                "blurb": spec.get("blurb", ""),
                "accuracy": spec.get("accuracy"),
                "ttft_ms": spec.get("ttft_ms"),
                "answer_words": spec.get("answer_words"),
                "provider": provider,
                # Lets the UI mark which models send questions off the device.
                "remote": provider != "local",
                "is_default": mid == self.default_id,
                "loaded": self.is_loaded(mid),
                "load_ms": round(self._load_ms.get(mid, 0.0)),
                "error": getattr(llm, "last_error", None) if llm else None,
                # A remote model with no key configured should not be offered.
                "available": (provider == "local"
                              or bool(config.groq_api_key())),
            })
        return out
