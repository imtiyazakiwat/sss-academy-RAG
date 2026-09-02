"""
Groq-hosted generation, exposing the same interface as LocalLLM so the model
registry and the app can treat it as one more selectable model.

Why it exists: Groq serves much larger open models than fit on this machine
(27B and 120B against a local 3B or 4B) at 280-1000+ tokens/sec, so answers are
both better and faster to finish. The prompt, retrieval, routing and answer
scrubbing are shared with the local path, so switching model changes only the
model.

What it costs, and why local stays the default:
  * Questions leave the device. The local models answer entirely offline.
  * The KV-cache reuse that makes local follow-ups ~240ms cannot apply, because
    the cache lives on Groq's side. Network round-trip becomes the floor.
  * The free tier is roughly 30 requests/minute shared across everyone using
    the key, which is thin for a classroom.
"""

import time

import config
from models.local_llm import (
    build_context,
    build_system_prompt,
    build_user_message,
    scrub_answer,
    scrub_line,
    wants_table,
)

# Reasoning models (gpt-oss, qwen3.6) emit a thinking block before the answer.
_THINK_CLOSE = "</think>"


class RemoteLLM:
    """Chat-completions backend for Groq. Mirrors LocalLLM's public surface:
    is_loaded, generate(..., stream=), warmup(), unload()."""

    def __init__(self, model_path=None, api_key=None, reasoning_effort=None):
        self.model_path = model_path or ""
        self.reasoning_effort = reasoning_effort
        self._client = None
        self.is_loaded = False
        self.last_error = None

        key = api_key or config.groq_api_key()
        if not key:
            self.last_error = (
                "No Groq API key. Put GROQ_API_KEY in .env or the environment."
            )
            print(f"Groq model '{self.model_path}' unavailable: {self.last_error}")
            return
        try:
            # The official SDK is required rather than raw urllib: Cloudflare
            # rejects urllib's default user-agent with error 1010 (HTTP 403).
            from groq import Groq
            self._client = Groq(api_key=key, timeout=config.GROQ_TIMEOUT_S)
            self.is_loaded = True
            print(f"Groq model ready: {self.model_path}")
        except Exception as e:
            self.last_error = f"Groq client init failed: {e}"
            print(self.last_error)

    # -- helpers ---------------------------------------------------------
    def _messages(self, question, context_chunks, style, mode):
        table = wants_table(question, context_chunks)
        system = build_system_prompt(style=style, mode=mode, table=table)
        context = build_context(context_chunks, question=question)
        user = build_user_message(question, context)
        return (
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            table,
        )

    def _kwargs(self, table, mode):
        if table:
            max_tokens = config.MAX_TOKENS_TABLE
        elif mode == "large":
            max_tokens = config.MAX_TOKENS_LARGE
        else:
            max_tokens = config.MAX_TOKENS
        kw = {
            "model": self.model_path,
            # Headroom for reasoning models: gpt-oss-120b spent an entire
            # 700-token budget on reasoning and returned empty content, so the
            # cap is raised and the thinking block is stripped below.
            "max_completion_tokens": int(max_tokens * config.GROQ_TOKEN_HEADROOM),
            "temperature": config.TEMPERATURE,
        }
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        return kw

    # -- generation ------------------------------------------------------
    def generate(self, question, context_chunks, stream=False,
                 style="grounded", mode="fast"):
        if not self.is_loaded:
            msg = "The online model is unavailable. Pick a local model instead."
            if stream:
                def dead():
                    yield msg, 0.0
                    yield "", 0.0
                return dead()
            return msg, 0.0, 0.0

        messages, table = self._messages(question, context_chunks, style, mode)
        kwargs = self._kwargs(table, mode)
        t0 = time.time()

        if stream:
            return self._stream(messages, kwargs, t0)

        try:
            r = self._client.chat.completions.create(messages=messages, **kwargs)
            text = r.choices[0].message.content or ""
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return self._failure(e), elapsed, elapsed
        text = _strip_reasoning(text)
        elapsed = (time.time() - t0) * 1000
        return scrub_answer(text), elapsed, elapsed

    def _stream(self, messages, kwargs, t0):
        """Yield (text_piece, ttft_ms), then ("", total_ms) as the sentinel.

        Same line-buffered scrubbing as the local path: the first line streams
        straight through so time-to-first-token is unaffected, later lines are
        held until complete so any line that talks about the material instead of
        answering can be dropped before the student sees it."""
        def gen():
            ttft = None
            buf = ""
            first_line_done = False
            in_reasoning = False
            try:
                completion = self._client.chat.completions.create(
                    messages=messages, stream=True, **kwargs)
                for chunk in completion:
                    piece = chunk.choices[0].delta.content or ""
                    if not piece:
                        continue

                    # Swallow a reasoning block if the model emits one.
                    if "<think>" in piece:
                        in_reasoning = True
                    if in_reasoning:
                        if _THINK_CLOSE in piece:
                            in_reasoning = False
                            piece = piece.split(_THINK_CLOSE, 1)[1]
                            if not piece:
                                continue
                        else:
                            continue

                    if ttft is None:
                        ttft = (time.time() - t0) * 1000

                    # First line streams straight through so TTFT is unaffected.
                    # Never yield an empty string mid-stream: that is the
                    # completion sentinel and would truncate the answer.
                    if not first_line_done:
                        if "\n" in piece:
                            head, _, buf = piece.partition("\n")
                            if head:
                                yield head, ttft
                            yield "\n", ttft
                            first_line_done = True
                        else:
                            yield piece, ttft
                        continue

                    buf += piece
                    while "\n" in buf:
                        line, _, buf = buf.partition("\n")
                        out = scrub_line(line)
                        if out:
                            yield out + "\n", ttft
                if buf:
                    out = scrub_line(buf)
                    if out:
                        yield out, ttft
            except Exception as e:
                yield self._failure(e), (ttft or (time.time() - t0) * 1000)
            yield "", (time.time() - t0) * 1000
        return gen()

    def _failure(self, err):
        """A student-facing message. Never leaks the key or a stack trace."""
        text = str(err)
        self.last_error = text
        low = text.lower()
        if "rate" in low or "429" in low:
            return ("The online model is busy right now (rate limit). "
                    "Try again shortly, or switch to a local model.")
        if "auth" in low or "api key" in low or "401" in low:
            return ("The online model is not configured correctly. "
                    "Use a local model for now.")
        print(f"Groq request failed: {text[:300]}")
        return ("Could not reach the online model. "
                "Check the connection, or switch to a local model.")

    # -- parity with LocalLLM -------------------------------------------
    def warmup(self):
        """Nothing to prefill remotely. One tiny call confirms the key works so
        the UI can show the model as ready rather than failing on first use."""
        if not self.is_loaded:
            return 0.0
        t0 = time.time()
        try:
            self._client.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": "ok"}],
                max_completion_tokens=1, temperature=0,
            )
        except Exception as e:
            self.last_error = str(e)
            print(f"Groq warmup failed for {self.model_path}: {str(e)[:200]}")
            self.is_loaded = False
        return (time.time() - t0) * 1000

    def unload(self):
        self._client = None
        self.is_loaded = False


def _strip_reasoning(text: str) -> str:
    if _THINK_CLOSE in text:
        text = text.split(_THINK_CLOSE, 1)[1]
    return text.strip()
