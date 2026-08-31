"""
Answer generator with confidence-based routing.

Routing:
  - score >= 0.90 : return the top chunk's content directly (no LLM) -> <30ms
  - 0.60 <= score < 0.90 : synthesize answer from top chunks via local LLM
  - score <  0.60 : return "This information is not available in the knowledge base."
"""

from config import LOW_CONFIDENCE, MIN_LEXICAL_FOR_WEAK_VECTOR, MIN_VECTOR_FOR_WEAK_LEXICAL
from models.local_llm import LocalLLM

NOT_AVAILABLE = "This information is not available in the knowledge base."


class AnswerGenerator:
    def __init__(self, llm: LocalLLM, low_conf=LOW_CONFIDENCE,
                 min_vector=MIN_VECTOR_FOR_WEAK_LEXICAL,
                 min_lexical=MIN_LEXICAL_FOR_WEAK_VECTOR):
        self.llm = llm
        self.low_conf = low_conf
        self.min_vector = min_vector
        self.min_lexical = min_lexical

    def generate(self, question, retrieved, top_score):
        """
        Returns (answer, mode, generation_ms).
        mode in {'generated', 'extracted', 'unsupported'}

        Routing:
          - If retrieval does not meaningfully match the question (low
            confidence and/or no keyword overlap with weak vector score),
            return the out-of-context message.
          - Otherwise synthesize a grounded answer from the top chunks.
            Fall back to the raw top chunk only if the LLM is unavailable.
        """
        top = retrieved[0] if retrieved else None
        if top is None:
            return {
                "answer": NOT_AVAILABLE,
                "mode": "unsupported",
                "generation_ms": 0.0,
            }

        lexical = top.get("lexical", 0.0)
        vector = top.get("vector_score", 0.0)

        # 1. Unsupported: no real relevance to the question
        if top_score < self.low_conf \
           or (vector < self.min_vector and lexical < self.min_lexical):
            return {
                "answer": NOT_AVAILABLE,
                "mode": "unsupported",
                "generation_ms": 0.0,
            }

        # 2. LLM unavailable -> return the best raw chunk (still grounded)
        if not self.llm or not self.llm.is_loaded:
            return {
                "answer": top["content"],
                "mode": "extracted",
                "generation_ms": 0.0,
            }

        # 3. Grounded LLM synthesis combines the top chunks into one answer
        text, ttft_ms, total_ms = self.llm.generate(question, retrieved)
        return {
            "answer": text,
            "mode": "generated",
            "generation_ms": total_ms,
            "ttft_ms": ttft_ms,
        }
