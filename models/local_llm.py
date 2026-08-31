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

# Strict grounding prompt template.
# NOTE: the off-topic rejection ("This information is not available...") is
# handled at the retrieval/confidence gate BEFORE the LLM is called, so we do
# NOT put that phrase in this prompt. Doing so primed the small 3B model to
# repeat it; removing it makes the model reliably answer from the (guaranteed
# on-topic) context instead of refusing.
SYSTEM_PROMPT = """You are an ETL interview assistant. The context below is the knowledge base section relevant to the user's question.

INSTRUCTIONS:
1. Answer SOLELY from CONTEXT. Never use outside knowledge or internet definitions.
2. Use CONTEXT to give a full, direct answer. Cover every point the question asks about that appears in CONTEXT, even when the wording differs.
3. Do not refuse or hedge: if CONTEXT discusses the concept, explain it using CONTEXT.
4. Reuse the EXACT technical terminology, SQL syntax, and test-case steps that appear verbatim in CONTEXT (e.g. surrogate key, ETL Effective Start Date, Active Row Flag='A'/'H', Version Number, MINUS queries).
5. Be CONCISE and COMPLETE: use short bullets, one line each. List every phase/type/step the question asks about, but do not write long paragraphs or repeat definitions. Finish the answer fully; do not stop partway. Aim for a compact answer that fully covers the question.

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
