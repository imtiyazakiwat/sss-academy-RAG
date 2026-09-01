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
    MAX_TOKENS_LARGE,
    MAX_CONTEXT_CHARS,
    TEMPERATURE,
    TOP_K_CONTEXT,
)

# Flexible grounding prompt template.
# NOTE: the LLM is instructed to use CONTEXT as the primary source of truth, but
# to fall back to its own expert knowledge if the context lacks the answer.
# It is explicitly told to avoid meta-phrases like "according to the context".
_BASE_INSTRUCTIONS = """You are an expert ETL and SQL interview prep assistant. You provide clear, practical, and highly accurate answers, matching the concise and informative tone of the provided CONTEXT notes.

RULES:
- Never use phrases like "based on the provided context", "in the given context", or "according to the text". Just answer the question directly and naturally.
- Use the CONTEXT as your primary source of truth, matching its exact terms, definitions, and SQL dialects.
- If the CONTEXT does not fully contain the answer, you MUST use your own expert knowledge to write queries, explain concepts, and answer creatively. Do not say the information is not available; provide a helpful and accurate answer anyway.
- Do not hedge or repeat the question.
- Always aim to provide a working solution, SQL query, or explanation for any question asked.

{detail}

CONTEXT:
{context}
"""

# "fast" mode: compact outline, one point per line, no padding.
FAST_DETAIL = """FORMAT: concise bullet list only. One bullet per point, one line each, no paragraphs, no intro/conclusion, no examples not in CONTEXT, and no padding."""

# "large" mode: detailed like notes, but still start with the main answer.
LARGE_DETAIL = """FORMAT: detailed like classroom notes, in short paragraphs and bullets. FIRST line must be: **Main Answer:** then a bold one-line summary of the direct answer. Then expand each point with CONTEXT's details, using its exact terms/SQL. Stay grounded; stop once all asked points are covered."""

PROMPTS = {
    "fast": _BASE_INSTRUCTIONS.format(detail=FAST_DETAIL, context="{context}"),
    "large": _BASE_INSTRUCTIONS.format(detail=LARGE_DETAIL, context="{context}"),
}


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

    def _build_prompt(self, question, context_chunks, mode="fast"):
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
        system = PROMPTS.get(mode, PROMPTS["fast"]).format(context=context_str)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, question, context_chunks, stream=False, mode="fast"):
        """Generate a full answer. Returns (text, ttft_ms, total_ms)."""
        if not self.is_loaded:
            return "⚠️ Local model not loaded.", 0.0, 0.0

        prompt = self._build_prompt(question, context_chunks, mode=mode)
        t0 = time.time()

        gen_kwargs = {
            "max_tokens": MAX_TOKENS_LARGE if mode == "large" else MAX_TOKENS,
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
