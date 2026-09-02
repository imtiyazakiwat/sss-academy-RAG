"""
Speech to text for the question box.

Two paths, because no single one is both instant and accurate:

  browser   The Web Speech API. Emits interim words while the student is still
            speaking, so text appears with no perceptible delay. On Safari and
            macOS it uses the system dictation engine and can work offline.
            Weak on domain vocabulary: it renders "SCD" as "SED" or "escede".

  whisper   whisper-large-v3-turbo on Groq. Measured 640-860 ms round trip for
            3-4 seconds of speech, and it transcribes "SCD Type 2",
            "de-normalized" and "BCNF" correctly, especially with the
            vocabulary prompt below. Not instant, but accurate.

The UI uses the browser path for live feedback and this one to correct the final
text, so the student sees words immediately and submits something accurate.

A note on the 100 ms target: a transcription round trip cannot meet it. Whisper
is ~700 ms and local Whisper on this machine is slower. What is under 100 ms is
the browser's interim result, which is why that path drives the live display.
"""

import io
import re
import time

import config

# Terms the notes use, with the spelling and casing they use. Speech engines get
# these wrong constantly, and a question that says "sed type two" will not
# retrieve the SCD Type 2 section.
DOMAIN_TERMS = [
    "SCD Type 1", "SCD Type 2", "SCD Type 3", "SCD",
    "OLTP", "OLAP", "ETL", "DWH", "BCNF", "1NF", "2NF", "3NF",
    "TRUNCATE", "DELETE", "DROP", "MERGE", "MINUS", "INTERSECT", "UNION ALL",
    "UNION", "DDL", "DML", "DCL", "TCL", "NVL2", "NVL", "NULLIF", "COALESCE",
    "DENSE_RANK", "ROW_NUMBER", "RANK", "LEAD", "LAG", "SUBSTR", "SYSDATE",
    "surrogate key", "primary key", "foreign key", "composite key", "unique key",
    "fact table", "dimension table", "star schema", "snow flake schema",
    "de-normalized", "normalization", "data warehouse", "data mart",
    "staging layer", "landing layer", "initial load", "incremental load",
    "defect life cycle", "bug life cycle", "regression testing", "smoke testing",
    "test case", "sprint retrospective", "sprint planning", "scrum master",
    "product backlog", "story point", "epic", "Informatica", "HP ALM", "TOAD",
    "SCOTT.EMP", "Oracle", "self join", "inner join", "outer join", "equi join",
    "cross join", "cartesian join", "non-equi join", "materialized view",
]

# What speech engines actually produce for those terms. Keys are matched
# case-insensitively on word boundaries.
MISHEARINGS = {
    r"\bsed\s+type\b": "SCD Type",
    r"\bescd\b|\bes\s?c\s?d\b|\besce?dee?\b": "SCD",
    r"\bsed\b(?=\s+(type|1|2|3|one|two|three))": "SCD",
    r"\bscd\s+type\s+one\b": "SCD Type 1",
    r"\bscd\s+type\s+two\b": "SCD Type 2",
    r"\bscd\s+type\s+three\b": "SCD Type 3",
    r"\bold\s?tp\b|\boh\s?l\s?t\s?p\b": "OLTP",
    r"\bold\s?ap\b|\boh\s?l\s?a\s?p\b|\bolap\b": "OLAP",
    r"\be\s?t\s?l\b": "ETL",
    r"\bdenormali[sz]ed\b|\bde\s+normali[sz]ed\b": "de-normalized",
    r"\bnormali[sz]ation\b": "normalization",
    r"\bsnowflake\b|\bsnow\s+flake\b": "snow flake",
    r"\bsurrogate\s+ki\b|\bsurrogate\s+case\b": "surrogate key",
    r"\btruncate?\b": "TRUNCATE",
    r"\bb\s?c\s?n\s?f\b|\bbecnf\b": "BCNF",
    r"\bfirst\s+normal\s+form\b": "1NF",
    r"\bsecond\s+normal\s+form\b": "2NF",
    r"\bthird\s+normal\s+form\b": "3NF",
    r"\bnull\s?if\b": "NULLIF",
    r"\bn\s?v\s?l\s?2\b|\bnvl\s+two\b": "NVL2",
    r"\bn\s?v\s?l\b": "NVL",
    r"\bdense\s+rank\b": "DENSE_RANK",
    r"\brow\s+number\b": "ROW_NUMBER",
    r"\bscott\s+dot\s+emp\b|\bscott\s+emp\b": "SCOTT.EMP",
    r"\binformatica?\b|\binfomatica\b": "Informatica",
    r"\bh\s?p\s+alm\b|\bhp\s+elm\b": "HP ALM",
    r"\btoad\b": "TOAD",
    r"\bequi\s+join\b|\bequal\s+join\b": "equi join",
    r"\bcartesian\b|\bcartesion\b": "cartesian",
    r"\bdata\s+ware\s?house\b": "data warehouse",
    r"\bdata\s+mart\b": "data mart",
    r"\bs\s?t\s?m\b": "STM",
}

# Passed to Whisper to bias decoding toward this vocabulary. Measured effect:
# "denormalized" became "de-normalized", matching the notes.
VOCAB_PROMPT = (
    "ETL testing and SQL interview: "
    + ", ".join(DOMAIN_TERMS[:40])
)


def correct_transcript(text: str) -> str:
    """Rewrite a transcript into the vocabulary the notes actually use.

    Retrieval matches on these exact terms, so "sed type two" reaching the
    retriever as-is finds nothing, while "SCD Type 2" finds the right section.
    """
    if not text:
        return ""
    out = text.strip()
    for pattern, replacement in MISHEARINGS.items():
        out = re.sub(pattern, replacement, out, flags=re.I)
    # Whisper returns a leading space and a trailing full stop on questions.
    out = out.strip()
    if out.endswith(".") and "?" not in out and _looks_like_question(out):
        out = out[:-1] + "?"
    return out


def _looks_like_question(text: str) -> bool:
    return bool(re.match(
        r"^\s*(what|which|why|how|when|who|where|explain|tell|write|list|give|"
        r"difference|compare|is|are|do|does|can)\b", text, re.I))


class Transcriber:
    """Groq-hosted Whisper. Same availability contract as RemoteLLM."""

    def __init__(self, model=None):
        self.model = model or config.TRANSCRIBE_MODEL
        self._client = None
        self.is_loaded = False
        self.last_error = None
        key = config.groq_api_key()
        if not key:
            self.last_error = "No Groq API key; online transcription disabled."
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=key, timeout=config.GROQ_TIMEOUT_S)
            self.is_loaded = True
        except Exception as e:
            self.last_error = f"Transcriber init failed: {e}"
            print(self.last_error)

    def transcribe(self, audio_bytes, filename="audio.webm", language=None):
        """Returns {text, raw, ms, error}."""
        if not self.is_loaded:
            return {"text": "", "raw": "", "ms": 0.0, "error": self.last_error}
        t0 = time.time()
        try:
            buf = io.BytesIO(audio_bytes)
            buf.name = filename
            kwargs = {
                "file": buf,
                "model": self.model,
                "response_format": "json",
                # Biases decoding toward the notes' vocabulary.
                "prompt": VOCAB_PROMPT,
                "temperature": 0,
            }
            if language:
                kwargs["language"] = language
            r = self._client.audio.transcriptions.create(**kwargs)
            raw = (getattr(r, "text", "") or "").strip()
        except Exception as e:
            msg = str(e)
            print(f"Transcription failed: {msg[:200]}")
            low = msg.lower()
            if "rate" in low or "429" in low:
                friendly = "Transcription is rate limited. Try again shortly."
            else:
                friendly = "Could not transcribe the audio. Type the question instead."
            return {"text": "", "raw": "", "ms": (time.time() - t0) * 1000,
                    "error": friendly}
        return {
            "text": correct_transcript(raw),
            "raw": raw,
            "ms": (time.time() - t0) * 1000,
            "error": None,
        }
