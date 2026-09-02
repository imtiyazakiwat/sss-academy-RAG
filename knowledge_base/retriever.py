"""
Hybrid retriever: BM25 keyword hits fused with FAISS vector hits via
Reciprocal Rank Fusion, reranked by a cross-encoder, then expanded from the
matched child fragment to its full parent section ("small-to-big").

Design notes
------------
* Query expansion is STATIC. An earlier version called the LLM to rewrite each
  query into keywords; measured at 685 ms per request on this machine, and it
  emitted degenerate output ("JOIN, JOIN, JOIN, ...") that diluted BM25 term
  statistics. A lookup table is faster, deterministic, and more accurate.

* Expansions map *interview phrasing* onto *this PDF's actual vocabulary*.
  That distinction matters: the notes say "2nd max salary", never "2nd highest
  salary"; they say "de-normalized", never "snowflake". Without the mapping,
  naturally-worded questions miss the notes that answer them.

* Retrieval matches small child chunks for precision, then hands the LLM the
  whole parent section, so multi-part topics (the seven join types, the normal
  forms, the SCD types) arrive complete.
"""

import math
import re
import time
from collections import defaultdict

from .embeddings import Embeddings
from .vector_store import VectorStore
from .bm25_index import BM25Index

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# Question phrasing -> terms that actually occur in the notes.
# Verified against the PDF text, not guessed.
QUERY_EXPANSIONS = [
    (r"\bhistoric(al)?\b", ["history", "SCD Type 2", "version", "flag"]),
    (r"\baddress\s+chang", ["SCD Type 2", "dimension change", "historical", "version"]),
    (r"\bscd\b", ["slowly changing dimension", "SCD Type 1", "SCD Type 2", "SCD Type 3"]),
    # The notes never write "highest"; they write "max".
    (r"\b(2nd|second)\s+(highest|largest|top)\b", ["2nd max salary", "dense_rank", "drank"]),
    (r"\b(nth|3rd|third)\s+(highest|largest)\b", ["dense_rank", "drank", "max salary"]),
    (r"\bhighest\b", ["max", "max salary", "dense_rank"]),
    (r"\blowest\b", ["min", "min salary"]),
    # The notes spell it "Snow Flake" as two words, so the one-word spelling
    # every student actually types must be mapped across.
    (r"\bsnowflakes?\b", ["snow flake", "snow flake schema", "star schema", "normalized"]),
    (r"\bstar\s+schema\b", ["star schema", "snow flake schema", "fact table"]),
    (r"\bdenormaliz|de-normaliz", ["de-normalized", "star schema", "dimension table"]),
    (r"\bnormaliz", ["normalization", "normal forms", "1NF", "2NF", "3NF", "BCNF"]),
    (r"\bsurrogate\b", ["surrogate key", "sequence", "primary key", "OLAP"]),
    # "auto generated id column" is how students describe a surrogate key. The
    # notes only ever call it that, so without this the query drifts to the
    # data-warehouse and data-mart sections instead.
    (r"\b(auto[\s-]?generat\w*|automatically\s+generat\w*)\b",
     ["surrogate key", "sequence", "generated automatically", "numeric"]),
    (r"\b(id|key)\s+column\b|\bunique\s+(id|identifier)\b",
     ["surrogate key", "primary key", "unique record"]),
    (r"\bfact\b", ["fact table", "measures", "foreign key", "granularity"]),
    (r"\bdimension\b", ["dimension table", "de-normalized", "SCD"]),
    (r"\btruncate\b", ["truncate", "delete", "drop", "DDL", "DML", "rollback"]),
    (r"\bduplicate", ["duplicate", "distinct", "group by", "having", "row_number"]),
    (r"\bjoins?\b", ["join", "inner join", "equi join", "outer join", "self join"]),
    (r"\bwindow\s+function|analytic", ["analytical function", "row_number", "rank", "dense_rank", "lead", "lag"]),
    (r"\bnull\b", ["NVL", "NVL2", "NULLIF", "COALESCE"]),
    (r"\bdefect|\bbug\b", ["defect life cycle", "severity", "priority", "status"]),
    (r"\bagile\b|\bscrum\b|\bsprint\b", ["agile methodology", "scrum master", "sprint", "backlog"]),
    # The notes describe each ceremony (sprint planning, retrospective, demo,
    # release) without ever using the word "ceremony".
    (r"\bceremon(y|ies)\b|\brituals?\b|\bmeetings?\s+in\s+agile\b",
     ["sprint planning", "sprint retrospective", "sprint demo", "release", "scrum"]),
    # The notes never write "standup" or "stand-up"; the role is "Scrum Master".
    (r"\bstand\s?-?up\b|\bdaily\s+(meeting|call|scrum)\b|\bwho\s+runs\b",
     ["scrum master", "sprint", "agile methodology", "supervisor"]),
    (r"\blife\s?cycle\b", ["defect life cycle", "bug life cycle", "status"]),
    (r"\btest\s+case|\btesting\b", ["test case", "testing", "validation", "expected result"]),
    (r"\bunix\b|\bshell\b|\bcommand line\b", ["unix commands", "grep", "chmod"]),
    (r"\bconstraint", ["constraints", "primary key", "foreign key", "check constraint", "not null"]),
    (r"\bview\b", ["view", "materialized view", "stores the data logically"]),
    # Only expand to the warehouse/mart sections when the question is ABOUT
    # them. Firing on any mention of "warehouse" pulled unrelated questions
    # ("auto generated id column in warehouse") into the Data Mart section.
    (r"\bolap\b|\boltp\b|\b(data\s+)?warehouse\s+(vs|versus|and)\b"
     r"|\bwhat\s+is\s+(a\s+)?(data\s+)?warehouse\b|\bdata\s+mart\b",
     ["OLTP", "OLAP", "data warehouse", "data mart"]),
    (r"\bset\s+operator|\bunion\b|\bminus\b|\bintersect\b", ["union", "union all", "intersect", "minus"]),
]

# Misspellings of domain terms -> correct form. Applied before search and
# before lexical scoring so typos still reach the right notes.
SPELL_CORRECTIONS = {
    "chek": "check", "chcek": "check", "ckecks": "checks", "ceck": "check",
    "contraint": "constraint", "contraints": "constraints",
    "contrain": "constraint", "constrint": "constraint",
    "cosnstraint": "constraint", "constrains": "constraints",
    "trunctae": "truncate", "truncat": "truncate", "truncatd": "truncated",
    "surrgote": "surrogate", "surogate": "surrogate", "surrogte": "surrogate",
    "foriegn": "foreign", "foreing": "foreign", "forign": "foreign",
    "primery": "primary", "primry": "primary", "seconary": "secondary",
    "distint": "distinct", "distnct": "distinct",
    "verion": "version", "verson": "version",
    "dimesion": "dimension", "demention": "dimension", "dimention": "dimension",
    "deffect": "defect", "difect": "defect",
    "extractin": "extraction", "extrction": "extraction",
    "normaliazation": "normalization", "normalisation": "normalization",
    "quey": "query", "querys": "query",
    "explin": "explain", "explaine": "explain",
    "defintion": "definition", "defination": "definition",
    "refernce": "reference", "refernece": "reference",
    "snwoflake": "snowflake", "scemas": "schemas", "schma": "schema",
    "databse": "database", "datsbase": "database", "datbase": "database",
    "herarchy": "hierarchy", "hierachy": "hierarchy",
    "wharehouse": "warehouse", "warehose": "warehouse",
    "agrigate": "aggregate", "aggregrate": "aggregate",
    "analitical": "analytical", "analyical": "analytical",
}

STOPWORDS = {
    "what", "is", "the", "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "with", "how", "why", "do", "does", "did", "are", "be", "you",
    "your", "me", "explain", "tell", "about", "write", "query", "queries",
    "when", "whats", "difference", "between", "please", "can", "i", "it",
    "give", "show", "list", "some", "any", "there", "define",
    # Interview question framing, not subject matter. Without these, "tell me
    # about yourself" looked like a question about an uncovered topic because
    # the notes write "my-self" rather than "yourself".
    "yourself", "yourselves", "myself", "ourselves", "tell", "describe",
    "brief", "briefly", "explaining", "mean", "means", "meaning",
}


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, bounded for speed."""
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
    """Replace known typo'd tokens with their correct domain terms."""
    return re.sub(
        r"[A-Za-z]+",
        lambda m: SPELL_CORRECTIONS.get(m.group(0).lower(), m.group(0)),
        query,
    )


def expand_query(query: str, max_variants=3) -> list:
    """Static retrieval variants. Applies EVERY matching expansion (the old
    code stopped at the first), capped so BM25 does not get diluted."""
    lower = query.lower()
    extra_terms = []
    for pattern, terms in QUERY_EXPANSIONS:
        if re.search(pattern, lower):
            extra_terms.extend(t for t in terms if t.lower() not in lower)

    variants = [query]
    if extra_terms:
        # Deduplicate, preserve order.
        seen, terms = set(), []
        for t in extra_terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                terms.append(t)
        joined = " ".join(terms[:10])
        variants.append(f"{query} {joined}")
        variants.append(joined)
    return variants[:max_variants]


def _stem(word: str) -> str:
    """Crude suffix stripper. Enough to match "activities" with "activity" or
    "ceremonies" with "ceremony" without pulling in a stemming dependency."""
    w = word
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _fuzzy_hits(q_token: str, content_tokens: set, content_lower: str) -> bool:
    """True if q_token occurs in the content, or is within edit distance of a
    content word (covers typos the dictionary does not list)."""
    if q_token in content_lower:
        return True
    max_dist = 1 if len(q_token) <= 4 else 2
    return any(
        len(w) >= 4 and _edit_distance(q_token, w) <= max_dist
        for w in content_tokens
    )


class Lexicon:
    """Document frequencies over the notes, used to tell a question's decisive
    words from its filler.

    Plain token-overlap scoring counts every word the same, which is how
    "How to read Python scripts?" scored 0.67 against the Unix commands section:
    *read* and *scripts* matched while *python*, the only word that decides what
    the question is about, appears nowhere in the notes. That pushed an
    uncovered topic into the grounded style, and the model's only honest move
    was to say the material does not cover it - a leak.

    Weighting by inverse document frequency makes the rare, topic-bearing words
    dominate, and terms absent from the corpus are reported so routing can send
    the question to the open style instead."""

    def __init__(self, documents):
        self.n_docs = max(1, len(documents))
        self.df = defaultdict(int)
        for doc in documents:
            for tok in set(re.findall(r"[a-z0-9]+", (doc or "").lower())):
                self.df[tok] += 1
        self.vocab = set(self.df)
        # Stems too, so an inflected question word still counts as covered:
        # the notes write "activities" and "my-self" where a student types
        # "activity" and "yourself".
        self.stems = {_stem(v) for v in self.vocab}
        # Every term any expansion can introduce. If a question triggers an
        # expansion whose terms the notes do use, the question's own wording
        # being absent is not evidence the topic is missing - the notes just
        # name it differently ("snowflake" -> "snow flake").
        self.expansion_terms = set()
        for _pattern, terms in QUERY_EXPANSIONS:
            for t in terms:
                self.expansion_terms.update(re.findall(r"[a-z0-9]+", t.lower()))

    def idf(self, token: str) -> float:
        # Absent terms get the highest weight: they are the most informative
        # thing a question can contain.
        df = self.df.get(token, 0)
        return math.log(1.0 + self.n_docs / (1.0 + df))

    def _in_vocab(self, token: str) -> bool:
        if token in self.vocab or _stem(token) in self.stems:
            return True
        # Tolerate typos so a misspelling is not mistaken for a new topic.
        if len(token) < 5:
            return False
        return any(len(v) >= 5 and _edit_distance(token, v) <= 1 for v in self.vocab)

    def unknown_share(self, question: str) -> float:
        """Share of the question's information carried by words the notes never
        use. Judging on presence alone was too blunt: "Star vs Snowflake" has
        one absent word out of four and is fully covered, while "read Python
        scripts" has one absent word that IS the question."""
        terms = self.question_terms(question)
        if not terms:
            return 0.0
        # Words that a synonym rule already knows about. If an expansion pattern
        # matched part of the question, the notes cover that idea under another
        # name ("snowflake" -> "snow flake"), so its absence is not a gap.
        aliased = self.aliased_terms(question)
        total = unknown = 0.0
        for t in terms:
            w = self.idf(t)
            total += w
            if self._in_vocab(t) or t in aliased:
                continue
            unknown += w
        return unknown / total if total else 0.0

    def aliased_terms(self, question: str) -> set:
        """Question words that a synonym rule maps onto the notes' own wording.

        Only the text each pattern actually matched counts, so triggering the
        joins rule does not excuse an unrelated unknown word elsewhere in the
        same question."""
        corrected = correct_query(question).lower()
        out = set()
        for pattern, terms in QUERY_EXPANSIONS:
            for m in re.finditer(pattern, corrected):
                if not m.group(0).strip():
                    continue
                # Only credit the alias if the notes really use the replacement.
                if any(w in self.vocab or _stem(w) in self.stems
                       for t in terms for w in re.findall(r"[a-z0-9]+", t.lower())):
                    out.update(re.findall(r"[a-z0-9]+", m.group(0)))
        return out

    def question_terms(self, question: str):
        corrected = correct_query(question)
        return [
            w for w in re.findall(r"[a-z0-9]+", corrected.lower())
            if w not in STOPWORDS and len(w) > 2
        ]

    def unknown_terms(self, question: str):
        """Content words of the question that the notes never use."""
        return [t for t in self.question_terms(question) if not self._in_vocab(t)]

    def weighted_coverage(self, question: str, content: str) -> float:
        """Share of the question's INFORMATION, not its word count, that the
        given text covers."""
        terms = self.question_terms(question)
        if not terms:
            return 0.0
        content_lower = (content or "").lower()
        content_tokens = set(re.findall(r"[a-z0-9]+", content_lower))
        total = hit = 0.0
        for t in terms:
            w = self.idf(t)
            total += w
            if _fuzzy_hits(t, content_tokens, content_lower):
                hit += w
        return hit / total if total else 0.0


def lexical_relevance(query: str, content: str) -> float:
    """Fraction of meaningful query tokens present in the text. Used to judge
    whether retrieval actually matched the question's subject."""
    corrected = correct_query(query)
    q_tokens = {
        w for w in re.findall(r"[a-z0-9]+", corrected.lower())
        if w not in STOPWORDS and len(w) > 1
    }
    if not q_tokens:
        return 0.0
    content_lower = content.lower()
    content_tokens = set(re.findall(r"[a-z0-9]+", content_lower))
    hits = sum(1 for w in q_tokens if _fuzzy_hits(w, content_tokens, content_lower))
    return hits / len(q_tokens)


class HybridRetriever:
    def __init__(self, embeddings: Embeddings, vector_store: VectorStore,
                 bm25: BM25Index, parents=None, vector_top_k=10,
                 bm25_top_k=10, final_top_k=3, rerank_candidates=12,
                 detail_weight=1.5, match_weight=1.2, coverage_weight=3.0,
                 lexicon=None,
                 cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25 = bm25
        self.parents = parents or []
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.rerank_candidates = rerank_candidates
        self.detail_weight = detail_weight
        self.match_weight = match_weight
        self.coverage_weight = coverage_weight
        self.lexicon = lexicon

        if CrossEncoder:
            print(f"Loading CrossEncoder '{cross_encoder_model}'...")
            self.cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)
        else:
            print("WARNING: sentence_transformers missing, reranking disabled.")
            self.cross_encoder = None

    # -- warmup ----------------------------------------------------------
    def warmup(self):
        """Force lazy models to load and run one throwaway pass, so the first
        real user question does not pay an ~9 s model-load cost."""
        t0 = time.time()
        self.embeddings.load_model()
        self.retrieve("warmup query for model initialisation")
        return (time.time() - t0) * 1000

    # -- internals -------------------------------------------------------
    def _parent_for(self, meta):
        pid = meta.get("parent_id")
        if pid is None or pid >= len(self.parents):
            return None
        return self.parents[pid]

    def retrieve(self, query):
        """Hybrid retrieve -> rerank -> expand children to parent sections.

        Returns (results, breakdown). Each result carries the PARENT section in
        `content` (what generation reads) and the matched fragment in
        `child_content` (what the UI can cite)."""
        normalized = correct_query(query)
        variants = expand_query(normalized)

        # --- vector search over all variants, max score per doc ---
        vec_docs, vec_latency, vec_candidates = {}, 0.0, 0
        for v in variants:
            q_emb = self.embeddings.encode_query(v)
            results, lat = self.vector_store.search(q_emb, self.vector_top_k)
            vec_latency += lat
            vec_candidates += len(results)
            for r in results:
                idx = r["index"]
                if r["score"] > vec_docs.get(idx, -1.0):
                    vec_docs[idx] = float(r["score"])

        # --- BM25 over all variants, max score per doc ---
        bm25_docs, bm25_latency = {}, 0.0
        for v in variants:
            hits, lat = self.bm25.search(v, self.bm25_top_k)
            bm25_latency += lat
            for h in hits:
                did = h["doc_id"]
                if h["score"] > bm25_docs.get(did, -1.0):
                    bm25_docs[did] = h["score"]

        # --- Reciprocal Rank Fusion ---
        combined = defaultdict(float)
        sources = defaultdict(dict)
        k = 60
        for rank, (idx, score) in enumerate(
                sorted(vec_docs.items(), key=lambda x: x[1], reverse=True)):
            combined[idx] += 1.0 / (k + rank + 1)
            sources[idx]["vector"] = score
        for rank, (idx, score) in enumerate(
                sorted(bm25_docs.items(), key=lambda x: x[1], reverse=True)):
            combined[idx] += 1.0 / (k + rank + 1)
            sources[idx]["bm25"] = score

        metadata = self.vector_store.metadata
        candidates = sorted(combined, key=combined.get, reverse=True)[:self.rerank_candidates]

        # --- score children, then aggregate to parent sections ---
        # Children are scored (a short fragment is what a cross-encoder judges
        # reliably), but selection happens per PARENT, because the parent is
        # what the model reads. Three signals combine:
        #
        #   best_child   how well the sharpest fragment matches the question
        #   n_matched    how many distinct fragments of that section matched;
        #                the section that truly owns a topic has several
        #   size         a section too small to hold a full answer should not win
        #
        # The last two matter because these notes state each topic twice: in
        # full (joins, pages 35-36, seven types) and as a one-line summary
        # table (page 57, 286 chars). Cross-encoders systematically prefer the
        # short dense text, so raw reranking picked the summary.
        rerank_latency = 0.0
        child_score = {}
        if self.cross_encoder and candidates:
            t0 = time.time()
            # Rerank against the EXPANDED query. Scoring the raw wording alone
            # lets surface word overlap win: "who runs the daily standup" scored
            # the self-introduction section "And my Daily Activities are" above
            # the Scrum Master section, because the notes never say "standup".
            rerank_query = variants[1] if len(variants) > 1 else normalized
            pairs = [
                [rerank_query, metadata[i].get("index_text") or metadata[i]["content"]]
                for i in candidates
            ]
            scores = self.cross_encoder.predict(pairs)
            rerank_latency = (time.time() - t0) * 1000
            child_score = {i: float(s) for i, s in zip(candidates, scores)}
        else:
            child_score = {i: combined[i] for i in candidates}

        agg = {}
        for idx in candidates:
            pid = metadata[idx].get("parent_id")
            slot = agg.setdefault(pid, {"best": -1e9, "idx": idx, "n": 0})
            slot["n"] += 1
            if child_score[idx] > slot["best"]:
                slot["best"] = child_score[idx]
                slot["idx"] = idx

        best_child = {pid: slot["idx"] for pid, slot in agg.items()}
        ranked = []
        for pid, slot in agg.items():
            parent = self.parents[pid] if pid is not None and pid < len(self.parents) else None
            body = parent["content"] if parent else metadata[slot["idx"]]["content"]
            size = len(body)
            # Does the section actually contain what the question asked about?
            # Without this, "what are the levels of testing" ranked two "Test
            # Case" sections with ZERO coverage of the question above the
            # "LEVELS OF TESTING" section, because they were longer and matched
            # more fragments. Coverage is idf-weighted, so a section holding the
            # rare, decisive words wins over one holding only common ones.
            coverage = (self.lexicon.weighted_coverage(query, body)
                        if self.lexicon is not None else 0.0)
            score = (
                slot["best"]
                + self.detail_weight * min(size / 2000.0, 1.0)
                + self.match_weight * math.log1p(slot["n"] - 1)
                + self.coverage_weight * coverage
            )
            ranked.append((pid, score))
        ranked.sort(key=lambda x: x[1], reverse=True)

        # --- build results, one entry per parent section ---
        results, seen_parents = [], {}
        for pid, score in ranked:
            idx = best_child[pid]
            meta = metadata[idx]
            parent = self._parent_for(meta)
            body = parent["content"] if parent else meta["content"]

            if pid in seen_parents:
                continue
            entry = {
                "topic": meta.get("topic", ""),
                "heading": (parent or meta).get("heading", ""),
                "page": meta.get("page"),
                "page_label": (parent or {}).get("page_label", f"page {meta.get('page')}"),
                "content": body,             # parent section -> fed to the LLM
                "child_content": meta["content"],   # matched fragment
                "header": meta.get("header", ""),
                # Sub-item label of the matched fragment, e.g. "d) Left Outer
                # Join". Its presence means retrieval pointed at one part of the
                # section rather than the section as a whole, which is what lets
                # context assembly trim safely.
                "label": meta.get("label", ""),
                "score": round(score, 4),
                "vector_score": round(sources[idx].get("vector", 0.0), 4),
                "bm25_score": round(sources[idx].get("bm25", 0.0), 4),
                "rrf_score": round(combined[idx], 4),
                "lexical": round(lexical_relevance(query, body), 3),
                "also_matched": [],
            }
            seen_parents[pid] = entry
            results.append(entry)
            if len(results) >= self.final_top_k:
                break

        breakdown = {
            "vector_latency_ms": round(vec_latency, 2),
            "bm25_latency_ms": round(bm25_latency, 2),
            "rerank_latency_ms": round(rerank_latency, 2),
            "vector_candidates": vec_candidates,
            "bm25_candidates": len(bm25_docs),
            "variants": len(variants),
            "reranked": len(candidates),
        }
        return results, breakdown
