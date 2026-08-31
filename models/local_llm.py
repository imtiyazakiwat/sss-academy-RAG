"""
Local generation model using the RAW (non-finetuned) Qwen 2.5 Instruct via MLX.
No LoRA adapter. RAG injects the knowledge base context at inference time,
so the model answers grounded in the PDF.

The model is loaded once at startup and kept resident in memory.
"""

import time
from mlx_lm import load, stream_generate, generate
from mlx_lm.sample_utils import make_sampler

from config import (
    LOCAL_MODEL,
    MAX_TOKENS,
    MAX_CONTEXT_CHARS,
    TEMPERATURE,
    TOP_K_CONTEXT,
)

# Strict grounding prompt template
SYSTEM_PROMPT = """You are an ETL interview assistant that answers ONLY from the SSS Academy classroom notes provided in CONTEXT.

HARD RULES:
1. Answer SOLELY from CONTEXT. Never use outside knowledge, general knowledge, or internet definitions.
2. This is a hard requirement: if CONTEXT does NOT contain enough to answer the question, reply EXACTLY and ONLY:
   "This information is not available in the knowledge base."
   Do NOT attempt to guess, generalize, or answer from prior knowledge when the specific information is missing.
3. Reuse the EXACT technical terminology, SQL syntax, and test-case steps that appear verbatim in CONTEXT (e.g. surrogate key, ETL Effective Start Date, Active Row Flag='A'/'H', Version Number, MINUS queries).
4. Give a short, direct answer with clear bullets only. Do not embellish or add fluff.

CONTEXT:
{context}
"""


class LocalLLM:
    def __init__(self, model_path=LOCAL_MODEL):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load()

    def _load(self):
        try:
            t0 = time.time()
            print(f"Loading local model '{self.model_path}' (RAW, no LoRA)...")
            self.model, self.tokenizer = load(
                self.model_path,
                tokenizer_config={"trust_remote_code": True},
            )
            load_ms = (time.time() - t0) * 1000
            print(f"✅ Raw model loaded in {load_ms:.0f} ms")
            self.is_loaded = True
        except Exception as e:
            print(f"Failed to load local model: {e}")
            self.is_loaded = False

    def _build_prompt(self, question, context_chunks):
        """Build the grounded generation prompt with a bounded context budget.

        Only the top chunk(s), truncated, are injected so that prefill (first
        token) stays fast on local MPS hardware.
        """
        context_blocks = []
        budget = MAX_CONTEXT_CHARS
        for c in (context_chunks or [])[:TOP_K_CONTEXT]:
            if budget <= 0:
                break
            topic = f"[{c['topic']}] " if c.get("topic") else ""
            page = f"(Page {c['page']})" if c.get("page") else ""
            content = (c.get("content") or "")[:budget]
            context_blocks.append(f"{topic}{page}\n{content}")
            budget -= len(content)

        context_str = "\n\n---\n\n".join(context_blocks)
        system = SYSTEM_PROMPT.format(context=context_str)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, question, context_chunks, stream=False):
        """Generate a full answer. Returns (text, ttft_ms, total_ms)."""
        if not self.is_loaded:
            return "⚠️ Local model not loaded.", 0.0, 0.0

        prompt = self._build_prompt(question, context_chunks)
        t0 = time.time()

        gen_kwargs = {
            "max_tokens": MAX_TOKENS,
            "sampler": make_sampler(temp=TEMPERATURE),
        }

        if stream:
            def gen():
                first_token_time = None
                for resp in stream_generate(
                    self.model, self.tokenizer,
                    prompt=prompt,
                    **gen_kwargs,
                ):
                    # Skip empty tokens: MLX emits occasional empty pieces
                    # mid-stream (e.g. whitespace). Breaking on "" would
                    # truncate the answer.
                    if not resp.text:
                        continue
                    if first_token_time is None:
                        first_token_time = (time.time() - t0) * 1000
                    yield resp.text, first_token_time
                total_ms = (time.time() - t0) * 1000
                yield "", total_ms  # true completion sentinel
            return gen()

        # Non-streaming
        text = generate(
            self.model, self.tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )
        ttft_ms = (time.time() - t0) * 1000
        return text.strip(), ttft_ms, ttft_ms

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
