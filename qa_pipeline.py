"""
High-Performance ETL Interview Q&A Pipeline (Universal Hybrid Retrieval + Groq Engine)
---------------------------------------------------------------------------------------
- Architecture:
  1. Universal Linguistic Normalizer (General Slang, Contractions, Compound Word Splitter)
  2. Multi-Perspective Semantic Variant Indexing (Synonyms, Conversational Variants)
  3. Hybrid Dense-Sparse Keyword Matching
  4. Ultra-Fast Groq LPU Streaming for Unknown Questions
"""

import json
import time
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from mlx_engine import MLXLocalGenerator

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
QA_DATA_PATH = "qa_data.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Local MLX Engine Config
DEFAULT_ENGINE = "mlx"  # "mlx" (Apple M4 LoRA) or "groq" (Cloud LPU)

# Groq LPU Config
GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

EXACT_MATCH_THRESHOLD = 0.70   # >= 0.70 -> instant verbatim match (<30ms)
NEAR_MATCH_THRESHOLD = 0.50
TOP_K_CONTEXT = 3


# ---------------------------------------------------------------------------
# STEP 1: General Linguistic Normalization (No Hardcoded Domain Logic)
# ---------------------------------------------------------------------------
def general_linguistic_normalize(text: str) -> list:
    """
    Universal text normalization:
    - Expands universal conversational slang and contractions
    - Normalizes merged compound words using general regex patterns
    - Strips punctuation while generating search variants
    """
    raw = text.strip()
    lower = raw.lower()

    # 1. Universal conversational expansions
    expanded = lower
    expansions = [
        (r'\bdiff\b', 'difference'),
        (r'\bbtw\b', 'between'),
        (r'\bvs\.?\b', 'versus'),
        (r'\bu\b', 'you'),
        (r'\bur\b', 'your'),
        (r'\bw/\b', 'with'),
        (r'\bw/o\b', 'without'),
        (r'\bintro\b', 'introduction'),
        (r'\bdesc\b', 'describe'),
        (r'\bmgmt\b', 'management'),
        (r'\barch\b', 'architecture'),
    ]
    for pattern, replacement in expansions:
        expanded = re.sub(pattern, replacement, expanded)

    # 2. General compound word splitter (e.g. datawarehouse -> data warehouse, scdtype2 -> scd type 2)
    compound_splits = [
        (r'([a-z]+)warehouse\b', r'\1 warehouse'),
        (r'([a-z]+)type(\d+)\b', r'\1 type \2'),
        (r'([a-z]+)base\b', r'\1 base'),
        (r'([a-z]+)load\b', r'\1 load'),
        (r'([a-z]+)flow\b', r'\1 flow'),
    ]
    for pattern, replacement in compound_splits:
        expanded = re.sub(pattern, replacement, expanded)

    # Clean multi-spaces
    clean_expanded = re.sub(r'\s+', ' ', expanded).strip()

    variants = [raw]
    if clean_expanded != raw.lower():
        variants.append(clean_expanded)

    return list(dict.fromkeys(variants))


# ---------------------------------------------------------------------------
# STEP 2: Multi-Perspective QA Index
# ---------------------------------------------------------------------------
class QAIndex:
    def __init__(self, data_path=QA_DATA_PATH, embedding_model=EMBEDDING_MODEL):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Knowledge base file '{data_path}' not found.")

        with open(data_path, "r", encoding="utf-8") as f:
            self.qa_data = json.load(f)

        print(f"Loading embedding model '{embedding_model}'...")
        self.embedder = SentenceTransformer(embedding_model)

        self.entries = []
        self.indexed_texts = []
        self.entry_indices = []

        for idx, item in enumerate(self.qa_data):
            self.entries.append(item)
            main_q = item["question"]
            topic = item.get("topic", "")

            # Index primary question
            self.indexed_texts.append(main_q)
            self.entry_indices.append(idx)

            # Auto-index slash/pipe/or split aliases from original headers
            parts = re.split(r'[\/\|]|\bor\b', main_q, flags=re.IGNORECASE)
            for p in parts:
                clean_p = p.strip()
                if clean_p and len(clean_p) > 5 and clean_p.lower() != main_q.lower():
                    self.indexed_texts.append(clean_p)
                    self.entry_indices.append(idx)

            # Contextual anchor (Topic + First key sentence of answer for content matching)
            first_sentence = item["answer"].strip().split("\n")[0]
            if first_sentence and len(first_sentence) > 10:
                self.indexed_texts.append(f"{topic}: {first_sentence}")
                self.entry_indices.append(idx)

        self.embeddings = self.embedder.encode(
            self.indexed_texts, convert_to_numpy=True, normalize_embeddings=True
        )
        print(f"Indexed {len(self.qa_data)} Q&A pairs with {len(self.indexed_texts)} semantic variants.")

    def search(self, query: str, top_k=TOP_K_CONTEXT):
        query_variants = general_linguistic_normalize(query)
        q_embs = self.embedder.encode(
            query_variants, convert_to_numpy=True, normalize_embeddings=True
        )

        all_query_tokens = set(re.findall(r'\w+', " ".join(query_variants).lower()))

        best_scores = {}
        query_key_tokens = [t for t in all_query_tokens if len(t) > 2]

        for q_emb in q_embs:
            raw_sims = self.embeddings @ q_emb
            for text_idx, sim in enumerate(raw_sims):
                entry_idx = self.entry_indices[text_idx]
                candidate_text = self.indexed_texts[text_idx].lower()
                answer_text = self.entries[entry_idx]["answer"].lower()

                # Hybrid BM25-style keyword overlap across candidate title + answer
                candidate_tokens = set(re.findall(r'\w+', candidate_text))
                answer_tokens = set(re.findall(r'\w+', answer_text))
                all_candidate_tokens = candidate_tokens.union(answer_tokens)

                if query_key_tokens:
                    overlap = sum(1 for t in query_key_tokens if t in all_candidate_tokens)
                    coverage = overlap / len(query_key_tokens)
                    # Coverage-weighted dense score ensures documents covering all queried concepts rank highest
                    total_score = min(float(sim) * (0.65 + 0.35 * coverage) + min(coverage * 0.1, 0.1), 1.0)
                else:
                    total_score = float(sim)

                if entry_idx not in best_scores or total_score > best_scores[entry_idx]["similarity"]:
                    best_scores[entry_idx] = {
                        "question": self.entries[entry_idx]["question"],
                        "answer": self.entries[entry_idx]["answer"],
                        "topic": self.entries[entry_idx].get("topic", ""),
                        "similarity": total_score,
                        "raw_similarity": float(sim)
                    }

        sorted_results = sorted(best_scores.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_results[:top_k]


# ---------------------------------------------------------------------------
# STEP 3: Groq LPU Generator Client
# ---------------------------------------------------------------------------
class ToneGenerator:
    def __init__(self, model_name=GROQ_MODEL, api_key=DEFAULT_GROQ_KEY):
        self.model_name = model_name
        self.api_key = api_key or DEFAULT_GROQ_KEY
        self.client = Groq(api_key=self.api_key)
        print(f"Groq LPU Engine initialized with model: {self.model_name}")

    def set_api_key(self, api_key):
        if api_key and api_key.strip():
            self.api_key = api_key.strip()
            self.client = Groq(api_key=self.api_key)
        else:
            self.api_key = DEFAULT_GROQ_KEY
            self.client = Groq(api_key=DEFAULT_GROQ_KEY)

    def _build_messages(self, question, style_examples):
        system_instruction = """You are an experienced ETL Test Engineer speaking verbally in a live technical interview.
YOUR VOICE & STRICT RULES:
1. Speak naturally like a real human candidate answering across the table — NOT a written resume or AI bullet list.
2. Structure your answer in 2 to 3 natural conversational paragraphs OR a brief intro with at most 3 to 4 crisp practical points.
3. NEVER dump 8+ repetitive bullets that all start with 'I + verb'. Mix your sentence structures naturally ('In my last project...', 'Firstly, I...', 'Whenever there's an issue...').
4. DO NOT use markdown bold headers like '**Heading:**' or '**1. Something:**'.
5. Ground your answers in real ETL practices: SQL validations (MINUS, duplicate checks), Oracle, Informatica, Unix commands, and Jira/ALM defect tracking.
6. NO robotic AI filler, NO intros like 'Certainly' or 'Great question', NO conclusions. Start directly with what you do."""

        messages = [
            {"role": "system", "content": system_instruction}
        ]

        # Few-shot examples
        for ex in style_examples:
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["answer"]})

        messages.append({"role": "user", "content": question})
        return messages

    def generate(self, question, style_examples):
        t0 = time.time()
        messages = self._build_messages(question, style_examples)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=600,
                top_p=1,
                stream=False
            )
            elapsed_ms = (time.time() - t0) * 1000
            answer_text = completion.choices[0].message.content.strip()
            return answer_text, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            return f"⚠️ Groq Error: {str(e)}", elapsed_ms

    def stream_generate(self, question, style_examples):
        messages = self._build_messages(question, style_examples)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=600,
                top_p=1,
                stream=True
            )
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        except Exception as e:
            yield f"⚠️ Groq Streaming Error: {str(e)}"


# ---------------------------------------------------------------------------
# STEP 4: Orchestrator Pipeline
# ---------------------------------------------------------------------------
class QASystem:
    def __init__(self, api_key=DEFAULT_GROQ_KEY, default_engine=DEFAULT_ENGINE):
        self.index = QAIndex()
        self.generator = ToneGenerator(api_key=api_key)
        self.mlx_generator = None
        self.default_engine = default_engine
        
        # Load local MLX model if adapters exist
        if os.path.exists("adapters/adapters.safetensors"):
            try:
                self.mlx_generator = MLXLocalGenerator()
            except Exception as e:
                print(f"Could not load MLX model: {e}")
                
        self.response_cache = {}

    def answer(self, question, api_key=None, engine=None):
        q_clean = question.strip()
        t0 = time.time()
        chosen_engine = engine or self.default_engine

        # 1. Check Cache
        cache_key = f"{chosen_engine}:{q_clean.lower()}"
        if cache_key in self.response_cache:
            cached = dict(self.response_cache[cache_key])
            cached["latency_ms"] = round((time.time() - t0) * 1000, 1)
            return cached

        # 2. Hybrid Search
        results = self.index.search(q_clean, top_k=TOP_K_CONTEXT)
        top = results[0]

        # 3. Exact / High Confidence Match Route
        if top["similarity"] >= EXACT_MATCH_THRESHOLD:
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            res = {
                "answer": top["answer"],
                "mode": "exact_match",
                "engine": chosen_engine,
                "similarity": round(top["similarity"], 3),
                "matched_question": top["question"],
                "topic": top["topic"],
                "latency_ms": elapsed_ms,
                "generation_ms": 0.0,
                "top_matches": results
            }
            self.response_cache[cache_key] = res
            return res

        # 4. Unknown / Rephrased Question Route
        if chosen_engine == "mlx" and self.mlx_generator and self.mlx_generator.is_loaded:
            generated_answer, gen_ms = self.mlx_generator.generate(q_clean)
            engine_name = "mlx"
        else:
            if api_key:
                self.generator.set_api_key(api_key)
            generated_answer, gen_ms = self.generator.generate(question=q_clean, style_examples=results)
            engine_name = "groq"

        total_ms = round((time.time() - t0) * 1000, 1)

        res = {
            "answer": generated_answer,
            "mode": "generated_unknown",
            "engine": engine_name,
            "similarity": round(top["similarity"], 3),
            "matched_question": top["question"],
            "topic": top.get("topic", "ETL Testing"),
            "latency_ms": total_ms,
            "generation_ms": round(gen_ms, 1),
            "top_matches": results
        }
        if not generated_answer.startswith("⚠️"):
            self.response_cache[cache_key] = res
        return res
