"""
Answer generator: decides HOW to answer, then generates.

Routing picks one of three answer styles. It no longer refuses questions.
Every question gets a real answer; grounding strength only decides whether the
model must stay inside the notes, reason from them, or fall back on its own
expertise while keeping the notes' voice.

  grounded  strong keyword/semantic overlap -> answer strictly from the notes
  scenario  applied or situational question -> apply the notes' mechanism
  open      weak overlap -> answer from expertise, in the notes' register

The previous implementation had a route() that always returned "generated",
leaving every threshold in config.py as dead code, and reported `confidence` as
a raw cross-encoder logit (observed 4.3 to 7.6) while the config comments
described a 0.5-1.1 scale. Confidence is now a calibrated 0-1 value.
"""

import math
import re

from config import (
    GROUNDED_LEXICAL,
    GROUNDED_VECTOR,
    GROUNDED_RERANK,
    OPEN_LEXICAL,
)
from knowledge_base.retriever import lexical_relevance
from models.local_llm import LocalLLM

# Phrasings that signal an applied/situational question rather than a
# definition request.
_SCENARIO_PATTERNS = re.compile(
    r"\b(what\s+(will\s+)?happens?|what\s+if|if\s+(a|an|the|my|customer|source)"
    r"|how\s+(would|will|do)\s+you|how\s+to\s+handle|suppose|scenario|"
    r"in\s+your\s+project|real\s+time|client\s+ask|interviewer\s+ask|"
    r"which\s+(one\s+)?would\s+you|when\s+(a|an|the)\s+\w+\s+(changes?|fails?)|"
    r"steps?\s+(to|you)|approach|troubleshoot|debug|validate\s+that)\b",
    re.I,
)


def calibrate(rerank_score: float) -> float:
    """Squash a cross-encoder logit into a 0-1 confidence.

    ms-marco-MiniLM emits unbounded logits; observed relevant matches on this
    corpus land around 4 to 8, irrelevant ones below 0. A logistic centred at 2
    maps that onto a readable 0-1 range."""
    try:
        return round(1.0 / (1.0 + math.exp(-(rerank_score - 2.0) / 2.0)), 4)
    except OverflowError:
        return 0.0 if rerank_score < 0 else 1.0


class AnswerGenerator:
    """Routing is model-independent, so one generator serves every model.
    The LLM to use is supplied per call (students pick a model per question)."""

    def __init__(self, llm: LocalLLM = None):
        self.llm = llm

    def route(self, question, retrieved):
        """Choose an answer style. Cheap: no generation, no LLM call.

        Returns {'style', 'confidence', 'lexical', 'reason'}.
        """
        if not retrieved:
            return {"style": "open", "confidence": 0.0, "lexical": 0.0,
                    "reason": "no retrieval hits"}

        top = retrieved[0]
        rerank = float(top.get("score") or 0.0)
        vector = float(top.get("vector_score") or 0.0)
        # Recompute against the parent section: the retriever's value is
        # measured on whatever body it returned, and routing should judge the
        # exact text the model will read.
        lexical = top.get("lexical")
        if lexical is None:
            lexical = lexical_relevance(question, top.get("content", ""))
        lexical = float(lexical)
        confidence = calibrate(rerank)

        well_grounded = (
            lexical >= GROUNDED_LEXICAL
            or vector >= GROUNDED_VECTOR
            or rerank >= GROUNDED_RERANK
        )
        is_scenario = bool(_SCENARIO_PATTERNS.search(question))

        if is_scenario and lexical >= OPEN_LEXICAL:
            style, reason = "scenario", "applied phrasing with usable context"
        elif well_grounded:
            style, reason = "grounded", (
                f"lexical={lexical:.2f} vector={vector:.2f} rerank={rerank:.2f}"
            )
        elif lexical >= OPEN_LEXICAL:
            style, reason = "scenario", "partial overlap, reason from context"
        else:
            style, reason = "open", (
                f"weak overlap (lexical={lexical:.2f}), answering from expertise"
            )

        return {
            "style": style,
            "confidence": confidence,
            "lexical": round(lexical, 3),
            "reason": reason,
        }

    def generate(self, question, retrieved, mode="fast", llm=None):
        """Route, then generate with `llm` (defaults to the one given at init)."""
        decision = self.route(question, retrieved)
        llm = llm or self.llm

        if not llm or not llm.is_loaded:
            # No model: hand back the best section verbatim rather than an
            # apology, so the student still gets the material.
            return {
                "answer": retrieved[0]["content"] if retrieved else "",
                "mode": "extracted",
                "style": decision["style"],
                "confidence": decision["confidence"],
                "generation_ms": 0.0,
                "ttft_ms": 0.0,
            }

        text, ttft_ms, total_ms = llm.generate(
            question, retrieved, style=decision["style"], mode=mode
        )
        return {
            "answer": text,
            "mode": "generated",
            "style": decision["style"],
            "confidence": decision["confidence"],
            "route_reason": decision["reason"],
            "generation_ms": total_ms,
            "ttft_ms": ttft_ms,
        }
