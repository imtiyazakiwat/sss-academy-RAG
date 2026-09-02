"""
Local generation via MLX, using the raw (non-finetuned) instruct model.
RAG supplies the classroom notes at inference time.

Two things here are deliberate:

1. Three answer STYLES, not one.
   The previous single prompt told the model both to match the notes exactly
   and to "answer creatively" from its own knowledge. Those instructions fight
   each other, and the model resolved the conflict by inventing generic
   phrasing ("Returns rows when there is a match in both tables") in place of
   the notes' own wording ("retrieve the matching rows ... used equal to (=)
   operator in a condition"). Grounding strength now selects one coherent
   instruction set instead.

2. The static instructions are a CACHEABLE PREFIX.
   Retrieved context used to be interpolated into the system message, which
   made every request's prefix unique and defeated prompt caching. Context now
   lives in the user message, so the system block is byte-identical across
   requests and its KV cache is reused via LRUPromptCache, cutting prefill work
   off the time-to-first-token.
"""

import re
import time

import mlx.core as mx
from mlx_lm import load, stream_generate, generate
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

# Imported as a module, not as names, so that values can be overridden at
# runtime (the benchmark swaps the model and forces greedy decoding). Importing
# the names directly would bind them at import time and silently ignore the
# override.
import config

# Voice anchors taken verbatim from the notes. They teach the model the
# register to imitate when it must answer something the notes do not cover:
# "By using <thing> we can <do>", a Syntax: line, then an Example:.
_STYLE_EXEMPLARS = """Study how these notes are written, and write in the same voice:

  Join
  By using the join function we can combine the row from two or multiple table based on common data type in attribute (column).

  DELETE
  By using this delete command we can delete the row from table.
  Syntax:
  DELETE FROM <TABLE_NAME> WHERE <CONDITION>;
  Example:
  DELETE FROM COLLAGE_1 WHERE SLNO=4;
"""

_COMMON_RULES = """You are an ETL Testing and SQL interview preparation assistant for SSS Academy students, answering from their classroom notes.

ABSOLUTE RULES:
- Never output the words "material", "KNOWLEDGE", or any tag such as <material>. Never mention notes, context, sources, documents, pages, or availability. Never say "based on the provided context", "according to the notes", "the context does not mention", or "this information is not available". Answer as a knowledgeable instructor speaking directly.
- Never repeat or rephrase the question back before answering.
- Never write "not listed", "not documented", "not mentioned", "not specified", "not available" or anything else about what you were given. Omit the point instead.
- Never invent an employer, client, city, date, tool or metric that is absent. For "your project" or "yourself", state only what is actually given.
- Use the vocabulary and SQL dialect of the notes: Oracle-style SQL, SCOTT.EMP table names, terms like de-normalized, surrogate key, initial and incremental load.
- COMPLETENESS DECIDES CORRECTNESS. Scan for every named item: numbered (1. 2. 3.), lettered (a) b) c)), or named types, forms, levels, stages and commands. EVERY one must appear with its name and a short meaning. Naming three of seven joins, or defining normalization without naming 1NF, 2NF, 3NF and BCNF, is a WRONG answer even if what you did write is true. Answer every part of the question."""

_STYLE_GROUNDED = """The material provided fully covers this question.

- Answer strictly from the material. Do not add tools, concepts, or claims that are absent from it.
- Reuse its exact terms, definitions, and SQL. Do not paraphrase technical wording into generic textbook phrasing. If it says "de-normalized", write de-normalized. If it names forms 1NF, 2NF, 3NF, BCNF, name them.
- If the material lists sub-types (a, b, c ... or 1, 2, 3), give EVERY one with its one-line meaning. Listing only some of them is a wrong answer."""

_STYLE_SCENARIO = """This is a scenario or applied question. The material provided contains the mechanism that answers it.

- Identify which concept from the material applies, name it using the notes' own term, then apply it to the specifics asked.
- Walk through what happens concretely, in order.
- Ground every technical claim in the material; supply only the reasoning that connects it to the scenario."""

_STYLE_OPEN = """The material provided is related background but does not directly answer this question.

- Answer it anyway, accurately and usefully, from your own expertise. Never refuse and never point out the gap.
- Where the material supplies a relevant term, definition, or convention, prefer it over your own wording so the answer stays consistent with what the student has been taught.
""" + _STYLE_EXEMPLARS

_FAST_FORMAT = """FORMAT: one line of direct answer, then a bullet list. One point per bullet, one line each. No preamble, no conclusion, no restating the question.

Brevity applies to each LINE, never to the NUMBER of points. Every named item in the answer gets its own bullet. Aim for 60-160 words: an answer under 30 words has almost certainly dropped something that was asked for."""

# Comparison questions ("difference between X and Y") are what interviewers ask
# most, and the notes themselves store these as two-column tables. A markdown
# table reads the way the student's own notes do, and it makes an omission
# obvious: a row with one side blank is visibly incomplete, whereas alternating
# prose bullets hide it.
_TABLE_FORMAT = """FORMAT: a markdown comparison table and nothing before it. Follow this example exactly, including the header row style:

| Aspect | DELETE | TRUNCATE |
|---|---|---|
| Command type | DML | DDL |
| WHERE clause | Supported | Not supported |
| Rollback | Can be rolled back | Cannot be rolled back |
| Performance | Low | High |

RULES FOR THE TABLE:
- The second and third header cells are the two real names being compared, taken from the material. Never write a placeholder, never write angle brackets, and never write the word material or any tag.
- The first column is a short aspect name, two or three words. Every aspect name must be DIFFERENT; never repeat one.
- Each row states how the two sides DIFFER, so the two cells must not be identical. If a statement applies to both equally, leave it out of the table.
- Only add a row when the material supports BOTH cells. If it does not cover an aspect, omit that row entirely. A short table of solid rows is correct; a padded one is wrong.
- A cell must never comment on the material itself. Phrases like "not documented", "not explicitly documented", "not mentioned", "not specified", "not stated", "not available", or "as documented in the notes" are forbidden anywhere in the table. If you were about to write one, delete that whole row.
- Never add a "Documentation" row or any row about what is or is not written down. Rows describe the two things being compared, nothing else.
- Keep each cell to a short phrase, not a sentence.
- Include every point of difference the material gives, and nothing that is not about these two things.
- If the material states a similarity, add one line under the table starting with "Similarity:".
- If the question ALSO asks something that is not a comparison (a definition, or which of the two comes first and why), answer that part first in one or two short bullets, then give the table. Never drop it.
- No preamble, no conclusion, no restating the question."""

_LARGE_FORMAT = """FORMAT: classroom-notes style. First line is **Main Answer:** followed by a one-line direct answer. Then expand with short bullets, and a Syntax: or Example: block with real SQL where it helps. Stop once everything asked is covered."""

_STYLES = {
    "grounded": _STYLE_GROUNDED,
    "scenario": _STYLE_SCENARIO,
    "open": _STYLE_OPEN,
}
_FORMATS = {"fast": _FAST_FORMAT, "large": _LARGE_FORMAT}

# "difference between X and Y", "X vs Y", "compare X and Y"
_COMPARISON_QUESTION = re.compile(
    r"\b(difference|differences|differ|compare|comparison|contrast|"
    r"distinguish|versus|vs\.?)\b|\bwhich\s+(one\s+)?is\s+better\b",
    re.I,
)


def is_comparison_question(question: str, heading: str = "") -> bool:
    """True when the question is asking for a comparison.

    The retrieved heading alone is NOT enough. Many topics are only written up
    inside a comparison table, so "What is a Surrogate Key and why is it used?"
    retrieves "PRIMARY KEY vs SURROGATE KEY" while asking about one thing.
    Treating that as a comparison answered a question the student never asked and
    dropped the "why is it used" part entirely. So a heading only triggers a
    table when the question names BOTH sides of it."""
    q = question or ""
    if _COMPARISON_QUESTION.search(q):
        return True

    low = f" {heading} ".lower()
    if " vs " not in low:
        return False
    left, _, right = low.partition(" vs ")
    ql = q.lower()

    def named(side):
        # Match on the side's distinctive words, ignoring generic filler, so
        # "star schema" matches "STAR SCHEMA" and "snow flake" matches
        # "snowflake" once punctuation and spacing are dropped.
        words = [w for w in re.findall(r"[a-z0-9]+", side)
                 if w not in {"the", "a", "an", "and", "or", "of", "normal",
                              "data", "table", "key", "schema", "query"}]
        if not words:
            return False
        squashed = re.sub(r"[^a-z0-9]", "", ql)
        return any(w in ql or w in squashed for w in words)

    return named(left) and named(right)


def material_supports_table(heading: str, content: str) -> bool:
    """True when the retrieved material really is a two-sided comparison.

    Asking for a table when the notes only hold two prose definitions makes the
    model pad: for "difference between severity and priority" the notes give
    just two sentences and one shared note, and the 3B filled the gap with rows
    like "Documentation | Not explicitly documented", which both invents
    content and talks about the material. Tabulate only where the notes
    themselves tabulate."""
    if " vs " in f" {heading} ".lower():
        return True
    # A rendered comparison always has paired "• Side: value" lines, so two
    # distinct labels each appearing at least twice means it is genuinely
    # two-sided.
    labels = re.findall(r"^\s*•\s*([^:\n]{2,40}):", content or "", re.M)
    counts = {}
    for lab in labels:
        counts[lab.strip().lower()] = counts.get(lab.strip().lower(), 0) + 1
    return sum(1 for n in counts.values() if n >= 2) >= 2


# A question is "broad" when it asks for a set rather than one fact, in which
# case the whole section must be sent or the answer will drop items.
_BROAD_QUESTION = re.compile(
    r"\b(types?\s+of|kinds?\s+of|forms?\s+of|list|all\s+the|every|"
    r"what\s+are|which\s+are|name\s+the|levels?\s+of|stages?\s+of|"
    r"steps?\s+of|categor|life\s?cycle|explain\s+the|different)\b",
    re.I,
)

# Plural technical nouns also imply a set is expected ("what are joins",
# "unix commands you used"), even without one of the phrases above.
_PLURAL_HINT = re.compile(
    r"\b(joins|keys|constraints|commands|operators|functions|schemas|"
    r"tables|types|forms|dimensions|defects|levels|stages|clauses)\b",
    re.I,
)



# Connectors that start a second, separate ask inside one question.
# Question marks already separate asks, so this only has to catch a second ask
# joined onto the first ("... and why?"). It requires an explicit conjunction or
# a comma: without that, the relative clause in "employees ... who joined last
# month" was mistaken for a second question.
_SECOND_ASK = re.compile(
    r"(?:,\s*|\s+)((?:and|also|then)\s+"
    r"(?:which|what|why|how|when|who|tell\s+me|explain|write|list|give)\b.*)$",
    re.I | re.S,
)


def question_parts(question: str):
    """Split a multi-part question into its separate asks.

    Interview questions in these notes routinely bundle two or three:
    "What is a fact and dimension table? Which table is loaded first in your
    project and why?" Both models answered only the first part every time, so
    the parts are extracted and listed back to the model as a checklist rather
    than left for it to notice."""
    q = (question or "").strip()
    if not q:
        return []

    # Sentence-level split first. A question mark is unambiguous; a full stop
    # counts only when the next sentence opens with a question word, so
    # "Explain the architecture of your project. How does data flow?" splits
    # while an ordinary abbreviation does not.
    chunks = [c.strip() for c in re.split(
        r"(?<=\?)\s+|(?<=\.)\s+(?=(?:which|what|why|how|when|who|tell|explain|write|list|give)\b)",
        q, flags=re.I) if c.strip()]

    parts = []
    for chunk in chunks:
        # Within a chunk, peel off a trailing second ask ("... and why?").
        rest = chunk
        m = _SECOND_ASK.search(rest)
        if m and len(m.group(1)) > 12 and len(rest[:m.start(1)].strip()) > 12:
            head = rest[:m.start(1)].strip(" ,.?")
            tail = m.group(1).strip(" ,.?")
            if head:
                parts.append(head)
            parts.append(tail)
        else:
            parts.append(rest.strip(" ,.?"))

    # Drop fragments too short to be a real ask.
    parts = [p for p in parts if len(p) > 8]
    return parts if len(parts) > 1 else []


def is_broad_question(question: str) -> bool:
    q = question or ""
    return bool(_BROAD_QUESTION.search(q) or _PLURAL_HINT.search(q))


def _words(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a, b):
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _focus_window(parent, child, limit, preamble_chars):
    """Section opening plus the neighbourhood of the matched fragment.

    The opening carries the definition ("Join / By using the join function we
    can combine the row from two or multiple table..."), and the matched
    fragment carries the specific detail the question asked about. Sending both
    keeps a narrow answer correct while staying far below the full section."""
    if not parent:
        return ""
    if len(parent) <= limit:
        return parent

    preamble = parent[:preamble_chars]
    if not child:
        return preamble + parent[preamble_chars:limit]

    # Locate the fragment by its first line, which survives chunking intact.
    probe = child.strip().split("\n")[0][:60]
    at = parent.find(probe) if probe else -1
    if at < 0:
        return parent[:limit]

    remaining = max(0, limit - len(preamble))
    # Start a little before the fragment so its own sub-heading is included.
    start = max(len(preamble), at - 120)
    window = parent[start:start + remaining]
    if start <= len(preamble):
        return parent[:limit]
    return f"{preamble}\n...\n{window}"


# Lines that talk about the material instead of answering. The prompt forbids
# these, but a 3B ignores the rule often enough to matter, so they are also
# removed deterministically. A student must never see the tool's plumbing.
_GAP_TALK = re.compile(
    r"not\s+(explicitly\s+|specifically\s+)?"
    r"(documented|mentioned|specified|stated|provided|available|listed|"
    r"included|found|covered|given)"
    r"|no\s+(relevant\s+)?(information|details?|data)\s+(is\s+)?"
    r"(available|provided|given|found)"
    r"|(the\s+)?(material|context|notes?|document|text)\s+(provided\s+)?"
    r"(does\s+not|doesn't|do\s+not|don't)"
    r"|based\s+on\s+the\s+(provided\s+)?(material|context)"
    r"|according\s+to\s+the\s+(material|context|notes)",
    re.I,
)


def scrub_line(line: str) -> str:
    """Drop a line that comments on the material rather than answering.

    Returns "" when the whole line should go. A table row is dropped outright; a
    prose line is dropped only if the gap talk is the substance of it, so a
    legitimate sentence that merely contains the word "provided" survives."""
    if not _GAP_TALK.search(line):
        return line
    stripped = line.strip()
    # Table rows and bullets are self-contained: remove the whole thing.
    if stripped.startswith("|") or stripped.startswith(("-", "*", "•")):
        return ""
    # Otherwise keep the line only if it still says something substantial once
    # the offending clause is removed.
    cleaned = _GAP_TALK.sub("", line).strip(" ,.;:-|")
    return line if len(cleaned) > 60 else ""


def scrub_answer(text: str) -> str:
    kept = [scrub_line(ln) for ln in (text or "").split("\n")]
    return "\n".join(ln for ln in kept if ln.strip() or ln == "").strip()


def build_context(context_chunks, question=""):
    """Assemble the material for the prompt, adaptively.

    Three rules, each earning its keep:

    1. Broad question -> whole section. "What are joins?" must see all seven
       join types, which is the full 2400-char section.
    2. Narrow question -> focused window. "What is a view?" needs the
       matched fragment and the section's opening line, not the whole
       section. This is where the prefill saving comes from, since prefill
       cost scales with context length.
    3. Skip near-duplicate sections. These notes teach several topics twice,
       and the top two hits are often the same material from two pages
       (measured Jaccard 0.99+), which would waste the second slot.
    """
    blocks, budget = [], config.MAX_CONTEXT_CHARS
    top_heading = context_chunks[0].get("heading", "") if context_chunks else ""
    # A comparison is answered as a table, and a table needs every row, so
    # it counts as broad and is never trimmed.
    broad = (is_broad_question(question)
             or is_comparison_question(question, top_heading))
    kept = []

    for c in (context_chunks or [])[:config.TOP_K_CONTEXT]:
        if budget <= 0:
            break
        parent = c.get("content") or ""
        if not parent:
            continue
        if any(_jaccard(parent, prev) >= config.CONTEXT_DEDUPE_JACCARD
               for prev in kept):
            continue

        # Trim only when it is safe: the question is not asking for a set,
        # the section is large, AND retrieval matched one LABELLED sub-item
        # rather than the section opening. That last condition matters:
        # "What is normalization?" reads narrow by phrasing but its answer
        # must name 1NF, 2NF, 3NF and BCNF, and its match is the section
        # opening, so it still gets the whole section.
        focusable = (
            not broad
            and len(parent) > config.CONTEXT_FOCUS_CHARS
            and bool((c.get("label") or "").strip())
        )
        if focusable:
            content = _focus_window(
                parent, c.get("child_content") or "",
                config.CONTEXT_FOCUS_CHARS, config.CONTEXT_PREAMBLE_CHARS,
            )
        else:
            content = parent
        content = content[:budget]
        if not content:
            continue

        heading = c.get("heading") or c.get("topic") or ""
        blocks.append(f"[{heading}]\n{content}" if heading else content)
        kept.append(parent)
        budget -= len(content)

    return "\n\n---\n\n".join(blocks)


def build_user_message(question, context):
    """The user turn: material first, then the question.

    Material comes first so two questions about the same section share a long
    token prefix and the KV cache can be reused. A multi-part question gets its
    parts listed back as a checklist, because both models otherwise answered
    only the first part of every bundled question.
    """
    parts = question_parts(question)
    if parts:
        asks = "\n".join(f"{i}. {p}" for i, p in enumerate(parts, 1))
        ask_block = (
            f"{question}\n\n"
            f"This question has {len(parts)} parts. Answer every one, each "
            f"under its own short heading:\n{asks}"
        )
    else:
        ask_block = question
    # The material is wrapped in a tag rather than labelled in plain text: a
    # label like "KNOWLEDGE:" gets echoed back as a heading in the answer.
    return (f"<material>\n{context}\n</material>\n\n{ask_block}"
            if context else ask_block)


def wants_table(question, context_chunks):
    """Whether this answer should be a comparison table."""
    top = context_chunks[0] if context_chunks else {}
    heading = top.get("heading", "")
    return (is_comparison_question(question, heading)
            and material_supports_table(heading, top.get("content", "")))


def build_system_prompt(style="grounded", mode="fast", table=False):
    """Static per (style, mode, table). Never contains retrieved text, so it
    stays byte-identical across requests and its KV cache can be reused."""
    return "\n\n".join([
        _COMMON_RULES,
        _STYLES.get(style, _STYLE_GROUNDED),
        _TABLE_FORMAT if table else _FORMATS.get(mode, _FAST_FORMAT),
    ])


class LocalLLM:
    def __init__(self, model_path=None, draft_model_path=None):
        model_path = model_path or config.LOCAL_MODEL
        draft_model_path = draft_model_path or config.DRAFT_MODEL
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_loaded = False
        # Holds prefilled KV caches: the per-style system prefixes, plus recent
        # full prompts so a follow-up on the same section reuses its prefill.
        # Byte-capped because each entry is tens of MB.
        self._prefix_cache = LRUPromptCache(
            max_size=config.PROMPT_CACHE_ENTRIES,
            max_bytes=config.PROMPT_CACHE_MAX_BYTES,
        )
        self._prefix_ids_by_system = {}
        self._load()

    def _load(self):
        try:
            t0 = time.time()
            print(f"Loading local model '{self.model_path}'...")
            self.model, self.tokenizer = load(
                self.model_path,
                tokenizer_config={"trust_remote_code": True},
            )
            print(f"Model loaded in {(time.time() - t0) * 1000:.0f} ms")
            if self.draft_model_path:
                # Optional speculative-decoding draft model: raises tokens/sec
                # during decode at a small extra memory cost.
                try:
                    self.draft_model, _ = load(self.draft_model_path)
                    print(f"Draft model loaded: {self.draft_model_path}")
                except Exception as e:
                    print(f"Draft model unavailable ({e}); continuing without it.")
                    self.draft_model = None
            self.is_loaded = True
        except Exception as e:
            print(f"Failed to load local model: {e}")
            self.is_loaded = False

    # -- prompt assembly -------------------------------------------------
    def _build_context(self, context_chunks, question=""):
        return build_context(context_chunks, question)

    def _render(self, messages, add_generation_prompt):
        """Apply the chat template with reasoning disabled where supported.

        Hybrid-reasoning models (Qwen3 / Qwen3.5) default to thinking mode: the
        template ends with an open `<think>` block, so the model emits a
        reasoning trace before the answer. For a sub-second interview assistant
        that is pure latency, so thinking is switched off when the template
        accepts the flag. Templates that do not accept it are unaffected."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except Exception:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def _static_prefix_ids(self, system):
        """Token ids of the longest reusable prefix for this system prompt.

        Rendering the system message on its own is not portable: Qwen3.5's
        template raises "No user query found in messages". Instead two full
        prompts are rendered that differ only in user content, and their longest
        common token prefix is taken. That is exactly the part every request with
        this system prompt shares, whatever the template does."""
        cached = self._prefix_ids_by_system.get(system)
        if cached is not None:
            return cached

        a = self.tokenizer.encode(self._render(
            [{"role": "system", "content": system},
             {"role": "user", "content": "AAAAAAAA"}],
            add_generation_prompt=True))
        b = self.tokenizer.encode(self._render(
            [{"role": "system", "content": system},
             {"role": "user", "content": "BBBBBBBB"}],
            add_generation_prompt=True))
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        prefix = a[:n]
        self._prefix_ids_by_system[system] = prefix
        return prefix

    def _build_prompt(self, question, context_chunks, style="grounded", mode="fast"):
        """Returns (full_token_ids, system_prefix_token_ids, is_table)."""
        # A comparison gets a table. The heading is consulted as well as the
        # question, so "Star Schema vs Snow Flake Schema" is tabulated even when
        # the student phrases it without the word "difference".
        # Tabulate only when the question asks for a comparison AND the material
        # actually holds one; otherwise the model invents rows to fill it.
        table = wants_table(question, context_chunks)
        system = build_system_prompt(style=style, mode=mode, table=table)
        context = build_context(context_chunks, question=question)

        user = build_user_message(question, context)

        full_text = self._render(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            add_generation_prompt=True,
        )
        full_ids = self.tokenizer.encode(full_text)
        prefix_ids = self._static_prefix_ids(system)
        return full_ids, prefix_ids, table

    def _cached_prefix(self, full_ids, prefix_ids):
        """Return (prompt_cache, tokens_still_to_process).

        Prefills and stores the system prefix the first time a given style is
        used, then reuses it for every later request with that style."""
        # LRUPromptCache keys its trie on a hashable identifier; the MLX Model
        # object itself is unhashable, so the model path is used.
        key = self.model_path
        cache, remaining = self._prefix_cache.fetch_nearest_cache(key, full_ids)
        if cache is not None:
            return cache, remaining

        cache = make_prompt_cache(self.model)
        # Only prefill if the rendered prefix really is a prefix of the full
        # prompt; chat templates can reorder blocks.
        if len(prefix_ids) > 8 and full_ids[:len(prefix_ids)] == prefix_ids:
            self.model(mx.array(prefix_ids)[None], cache=cache)
            mx.eval([c.state for c in cache])
            self._prefix_cache.insert_cache(
                key, prefix_ids, cache, cache_type="system"
            )
            cache, remaining = self._prefix_cache.fetch_nearest_cache(key, full_ids)
            return cache, remaining
        return cache, full_ids

    # -- generation ------------------------------------------------------
    def generate(self, question, context_chunks, stream=False,
                 style="grounded", mode="fast"):
        """Generate an answer. Returns (text, ttft_ms, total_ms), or a
        generator of (text_piece, ttft_ms) when stream=True."""
        if not self.is_loaded:
            return "Local model not loaded.", 0.0, 0.0

        full_ids, prefix_ids, table = self._build_prompt(
            question, context_chunks, style=style, mode=mode
        )
        cache, remaining = self._cached_prefix(full_ids, prefix_ids)

        if table:
            max_tokens = config.MAX_TOKENS_TABLE
        elif mode == "large":
            max_tokens = config.MAX_TOKENS_LARGE
        else:
            max_tokens = config.MAX_TOKENS
        gen_kwargs = {
            "max_tokens": max_tokens,
            "sampler": make_sampler(temp=config.TEMPERATURE),
            "prompt_cache": cache,
        }
        # A mild repetition penalty. Greedy decoding has no way out of a loop,
        # and a 3B model building a long comparison table can get stuck emitting
        # the same row ("| Data consistency | Single stream | Multiple streams |")
        # until it hits the token limit. The penalty is small so it does not
        # discourage the legitimate repetition of a term across table rows.
        if config.REPETITION_PENALTY and config.REPETITION_PENALTY != 1.0:
            gen_kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=config.REPETITION_PENALTY,
                repetition_context_size=config.REPETITION_CONTEXT,
            )
        if self.draft_model is not None:
            gen_kwargs["draft_model"] = self.draft_model

        t0 = time.time()

        if stream:
            def gen():
                ttft = None
                produced = []
                # The first line streams token by token so time-to-first-token
                # is unaffected. After that, output is buffered a line at a time
                # so a line that talks about the material can be dropped before
                # the student sees it - which is impossible once tokens are out.
                buf = ""
                first_line_done = False
                for resp in stream_generate(
                    self.model, self.tokenizer, prompt=remaining, **gen_kwargs
                ):
                    produced.append(resp.token)
                    # MLX emits occasional empty pieces mid-stream; skipping
                    # them (rather than breaking) avoids truncating answers.
                    if not resp.text:
                        continue
                    if ttft is None:
                        ttft = (time.time() - t0) * 1000

                    # The first line goes out token by token so TTFT is
                    # unaffected. NOTHING empty may be yielded mid-stream: the
                    # empty string is the completion sentinel, and yielding it
                    # here truncated every answer to its first line.
                    if not first_line_done:
                        if "\n" in resp.text:
                            head, _, buf = resp.text.partition("\n")
                            if head:
                                yield head, ttft
                            yield "\n", ttft
                            first_line_done = True
                        else:
                            yield resp.text, ttft
                        continue

                    buf += resp.text
                    while "\n" in buf:
                        line, _, buf = buf.partition("\n")
                        out = scrub_line(line)
                        if out:
                            yield out + "\n", ttft
                if buf:
                    out = scrub_line(buf)
                    if out:
                        yield out, ttft
                self._remember(full_ids, produced, cache)
                yield "", (time.time() - t0) * 1000  # completion sentinel
            return gen()

        text = generate(self.model, self.tokenizer, prompt=remaining, **gen_kwargs)
        self._remember(full_ids, self.tokenizer.encode(text), cache)
        elapsed = (time.time() - t0) * 1000
        return scrub_answer(text), elapsed, elapsed

    def _remember(self, full_ids, generated_ids, cache):
        """Store the used cache so the next question about the same section
        skips re-prefilling that section.

        The prompt is laid out system -> material -> question, so two different
        questions that retrieve the same section share a long token prefix. The
        cache is keyed on prompt+completion (what it actually holds); the trie
        then matches the shared prefix on a later request and trims the rest.
        For an interview-prep tool, where students ask several questions about
        one topic in a row, this turns the second question's prefill into
        almost nothing."""
        try:
            self._prefix_cache.insert_cache(
                self.model_path, list(full_ids) + list(generated_ids or []), cache
            )
        except Exception as e:
            # Caching is an optimisation; a failure must not break answering.
            print(f"prompt-cache insert skipped: {e}")

    def warmup(self):
        """Prefill and store each style's system prefix, and force MLX kernel
        compilation, so the first real request pays neither cost."""
        if not self.is_loaded:
            return 0.0
        t0 = time.time()
        ctx = [{"heading": "Warmup", "content": "Warmup section text."}]
        # The table variant is warmed too ("difference between X and Y" is one
        # of the commonest interview questions), so the first comparison does
        # not pay for prefilling a cold system prefix.
        # The heading "A vs B" is what makes material_supports_table() true, so
        # this really does build and cache the table system prefix.
        table_ctx = [{"heading": "A vs B", "content": "A vs B\nWarmup row."}]
        variants = [
            ("grounded", "warmup", ctx),
            ("scenario", "warmup", ctx),
            ("open", "warmup", ctx),
            ("grounded", "difference between a and b", table_ctx),
        ]
        for style, probe, probe_ctx in variants:
            full_ids, prefix_ids, _ = self._build_prompt(
                probe, probe_ctx, style=style, mode="fast"
            )
            cache, remaining = self._cached_prefix(full_ids, prefix_ids)
            for _ in stream_generate(
                self.model, self.tokenizer, prompt=remaining,
                max_tokens=1, prompt_cache=cache,
            ):
                break
        return (time.time() - t0) * 1000

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_loaded = False
