"""
Hybrid retriever: merges BM25 keyword hits with FAISS vector hits,
applies query expansion, then reranks to a final top-k.
"""

import re
from collections import defaultdict

from .embeddings import Embeddings
from .vector_store import VectorStore
from .bm25_index import BM25Index


# Query expansion map: surfacing hidden ETL concepts from paraphrased questions.
# Used ONLY for retrieval, never exposed to the model or user.
QUERY_EXPANSIONS = [
    # (regex pattern, expansion terms)
    (r"\bh?istoric(al)?\b", ["history", "SCD", "SCD Type 2", "version"]),
    (r"\baddress chang", ["SCD", "SCD Type 2", "dimension change", "historical"]),
    (r"\bchang(es|ed|ing)?\b", ["SCD", "SCD Type", "dimension change"]),
    ("scd", ["slowly changing dimension", "SCD Type 1", "SCD Type 2", "SCD Type 3"]),
    (r"\bfact\b", ["fact table", "measures", "foreign key"]),
    (r"\bsurrogate\b", ["surrogate key", "OLAP", "sequence", "primary key"]),
    (r"\btruncate\b", ["truncate", "delete", "drop", "DDL", "high water mark"]),
    (r"\b2nd highest", ["2nd highest salary", "Nth highest", "DENSE_RANK", "MAX"]),
    ("jointable", ["JOIN", "SQL join"]),
    (r"\bnormaliz", ["normalization", "normal forms", "1NF", "2NF", "3NF", "BCNF"]),
]


# Common misspellings of domain terms -> correct form. Applied to the query
# BEFORE vector/BM25 search and lexical scoring so typos still retrieve the
# right notes. Used ONLY for retrieval, never exposed to the user.
SPELL_CORRECTIONS = {
    "chek": "check", "chcek": "check", "ckecks": "checks", "ceck": "check",
    "contraint": "constraint", "contraints": "constraints", "contrain": "constraint",
    "constrint": "constraint", "cosnstraint": "constraint", "constrains": "constraints",
    "trunctae": "truncate", "truncat": "truncate", "truncatd": "truncated",
    "surrgote": "surrogate", "surrgote": "surrogate", "surrogate": "surrogate",
    "foriegn": "foreign", "foreing": "foreign", "forign": "foreign",
    "primery": "primary", "seconary": "secondary",
    "distinct": "distinct", "distint": "distinct",
    "verion": "version", "verson": "version",
    "dimesion": "dimension", "demention": "dimension",
    "defect": "defect", "deffect": "defect",
    "loading": "loading", "extractin": "extraction",
    "normalization": "normalization", "normaliazation": "normalization",
    "quey": "query", "queries": "query", "querys": "query",
    "explain": "explain", "explin": "explain", "explaine": "explain",
    "defintion": "definition", "defination": "definition",
    "refernce": "reference", "refernece": "reference",
    "snowflake": "snowflake", "snwoflake": "snowflake",
    "schem": "schema", "scemas": "schemas",
    "databse": "database", "datsbase": "database",
    "hierarchy": "hierarchy", "herarchy": "hierarchy",
    "subset": "subset", "subsetting": "subsetting",
}


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (small, bounded helper for fuzzy token matching)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def correct_query(query: str) -> str:
    """Replace known typo'd tokens with their correct domain terms.
    Preserves original spacing/punctuation."""
    return re.sub(
        r"[A-Za-z]+",
        lambda m: SPELL_CORRECTIONS.get(m.group(0).lower(), m.group(0)),
        query,
    )


def _fuzzy_hits(q_token: str, content_lower: str) -> bool:
    """True if q_token appears in content, or is within edit distance of a
    content word (handles typos the dictionary doesn't cover)."""
    if q_token in content_lower:
        return True
    max_dist = 1 if len(q_token) <= 4 else 2
    content_tokens = re.findall(r"[a-z0-9]+", content_lower)
    return any(
        len(w) >= 4 and _edit_distance(q_token, w) <= max_dist
        for w in content_tokens
    )



STOPWORDS = {
    "what", "is", "the", "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "with", "how", "why", "do", "does", "did", "are", "be", "you",
    "your", "me", "explain", "tell", "about", "write", "query", "queries",
    "when", "whats", "difference", "between", "please", "can", "i", "it",
}


def _lexical_relevance(query: str, content: str) -> float:
    """Fraction of meaningful query tokens that appear in the chunk.
    Exact-term interview topics (SCD, Fact Table, Truncate, PK/FK) carry
    strong weight because they are distinctive technical anchors.

    Typos are tolerated via spelling correction + fuzzy (edit-distance) match.
    """
    corrected = correct_query(query)
    q_tokens = {w for w in re.findall(r"[a-z0-9]+", corrected.lower()) if w not in STOPWORDS}
    if not q_tokens:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for w in q_tokens if _fuzzy_hits(w, content_lower))
    return hits / len(q_tokens)


def expand_query(query: str) -> list:
    """Return a list of retrieval query variants (for both BM25 and vector)."""
    lower = query.lower()
    variants = [query]
    for pattern, terms in QUERY_EXPANSIONS:
        if re.search(pattern, lower):
            variants.append(" ".join(terms))
            variants.append(f"{query} {' '.join(terms)}")
            break  # one conceptual expansion is enough to avoid dilution
    return variants


class HybridRetriever:
    def __init__(self, embeddings: Embeddings, vector_store: VectorStore,
                 bm25: BM25Index, vector_top_k=10, bm25_top_k=10, final_top_k=5):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25 = bm25
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query):
        """Run hybrid retrieval. Returns (final_results, breakdown).
        final_results: list of {topic, page, content, score}
        breakdown: {vector_score, bm25_score, type}"""
        # Correct obvious misspellings first so both vector and BM25 search
        # match the intended terms ("chek" -> "check", "contraint" -> ...).
        normalized = correct_query(query)
        variants = expand_query(normalized)

        # --- Vector search across ALL variants, taking max score per doc ---
        vec_docs = {}
        vec_candidates = 0
        vec_latency = 0.0
        for v in variants:
            q_emb = self.embeddings.encode_query(v)
            results, lat = self.vector_store.search(q_emb, self.vector_top_k)
            vec_latency += lat
            vec_candidates += len(results)
            for r in results:
                idx = r["index"]
                vec_docs.setdefault(idx, {"score": -1.0})
                if r["score"] > vec_docs[idx]["score"]:
                    vec_docs[idx] = {
                        "score": float(r["score"]),
                        "topic": r["topic"],
                        "page": r["page"],
                        "content": r["content"],
                    }

        # --- BM25 search across all variants, taking max per doc ---
        bm25_docs = {}
        bm25_latency = 0.0
        for v in variants:
            hits, lat = self.bm25.search(v, self.bm25_top_k)
            bm25_latency += lat
            for h in hits:
                doc_id = h["doc_id"]
                if doc_id not in bm25_docs or h["score"] > bm25_docs[doc_id]:
                    bm25_docs[doc_id] = h["score"]

        # --- Merge & rerank ---
        combined = defaultdict(float)
        sources = {}

        for idx, vec in vec_docs.items():
            combined[idx] += 0.7 * max(float(vec["score"]), 0.0)
            sources.setdefault(idx, {})["vector"] = float(vec["score"])

        if bm25_docs:
            max_bm25 = max(bm25_docs.values()) or 1.0
            for doc_id, score in bm25_docs.items():
                norm = score / max_bm25
                combined[doc_id] += 0.5 * norm
                sources.setdefault(doc_id, {})["bm25"] = norm

        # Convert back to results with metadata
        metadata = self.vector_store.metadata
        final = []
        for idx in sorted(combined, key=lambda i: combined[i], reverse=True)[:self.final_top_k]:
            meta = metadata[idx]
            final.append({
                "topic": meta["topic"],
                "page": meta["page"],
                "content": meta["content"],
                "score": round(combined[idx], 4),
                "vector_score": round(sources[idx].get("vector", 0.0), 4),
                "bm25_score": round(sources[idx].get("bm25", 0.0), 4),
                "lexical": round(_lexical_relevance(query, meta["content"]), 3),
            })

        breakdown = {
            "vector_latency_ms": round(vec_latency, 2),
            "bm25_latency_ms": round(bm25_latency, 2),
            "vector_candidates": vec_candidates,
            "bm25_candidates": len(bm25_docs),
        }
        return final, breakdown
