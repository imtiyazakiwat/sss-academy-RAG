"""
Hierarchical (parent/child) chunker for the SSS Academy classroom notes.

Why this design
---------------
The notes are structured teaching material: a bare topic heading ("Join"),
then an enumeration of sub-items ("a) Inner join ... g) Self join"), then one
labelled block per sub-item with bullets and SQL examples.

Retrieval needs SMALL units to match precisely, but generation needs the WHOLE
section to answer completely. Asking "what are joins" must surface all seven
join types, which live in different sub-blocks spanning two PDF pages.

So we emit two levels:

  parent  = a full topic section (definition + every sub-item), page-spanning
  child   = a small retrievable unit inside that section

Children are embedded and indexed. At generation time each child hit is
expanded to its parent ("small-to-big" retrieval). Every child also carries a
deterministic contextual header ("Section: Join > d) Left Outer Join") so the
fragment is self-describing to both BM25 and the embedding model.

Two bugs in the previous implementation are fixed here:

1. Heading detection matched a topic anchor ANYWHERE in any line under 80
   chars, so ordinary SQL body lines ("from departments D join employees e")
   became section boundaries and shredded the notes into 356 fragments.
   Headings are now required to be standalone labels and are rejected outright
   if they look like SQL.
2. Chunks were flushed on every page change, splitting concepts that continue
   across pages (Join runs 35 -> 36). Sections now span pages freely.
"""

import re

# Topic anchors. These only apply when the anchor dominates the line (see
# `_anchor_is_standalone`), never when it appears inside running prose or SQL.
TOPIC_ANCHORS = [
    "SCD TYPE", "SLOWLY CHANGING", "FACT TABLE", "DIMENSION TABLE",
    "NORMALIZATION", "NORMAL FORM", "1NF", "2NF", "3NF", "BCNF",
    "PRIMARY KEY", "SURROGATE KEY", "FOREIGN KEY", "COMPOSITE KEY",
    "STAR SCHEMA", "SNOWFLAKE SCHEMA", "DATA WAREHOUSE", "DATA MART",
    "DATA LAKE", "STAGING", "ETL", "ELT",
    "TRUNCATE", "DELETE", "DROP", "JOIN", "UNION", "INTERSECT", "MINUS",
    "ROW_NUMBER", "RANK", "DENSE_RANK", "LEAD", "LAG",
    "ANALYTICAL FUNCTION", "WINDOW FUNCTION", "AGGREGATE FUNCTION",
    "AGILE", "SCRUM", "SPRINT", "EPIC", "USER STORY", "BACKLOG",
    "TEST CASE", "TESTING", "SMOKE TESTING", "REGRESSION",
    "DEFECT", "BUG", "SDLC", "STLC", "SEVERITY", "PRIORITY",
    # Multi-word forms listed explicitly so the anchor covers enough of the line
    # to pass the dominance test: "Defect life cycle or Bug life cycle." is a
    # real heading in these notes but "DEFECT" alone is only 20% of it.
    "DEFECT LIFE CYCLE", "BUG LIFE CYCLE", "SCRUM MASTER", "SPRINT RETROSPECTIVE",
    "DDL", "DML", "DCL", "TCL", "CONSTRAINT",
    "NVL", "NVL2", "NULLIF", "COALESCE", "DECODE",
    "VIEW", "MATERIALIZED VIEW", "INDEX", "SUBQUERY",
    "UNIX", "DUPLICATE", "HIGHEST", "SELF INTRODUCTION", "ARCHITECTURE",
    "OLTP", "OLAP", "GROUP BY", "HAVING", "CASE",
    # Single-word SQL command topics. Listed explicitly because a lone word is
    # otherwise rejected as a heading (see is_section_heading) to stop column
    # names from a DESC listing (REGNUMBER, STUDENTNAME, BRANCH) becoming
    # sections.
    "INSERT", "UPDATE", "ALTER", "RENAME", "REVOKE", "GRANT",
    "COMMIT", "ROLLBACK", "SAVEPOINT", "PURGE", "MERGE",
]

# A line containing any of these is body text or SQL, never a heading.
_SQL_MARKERS = re.compile(
    r"\b(select|from|where|group\s+by|order\s+by|having|set|values|update|"
    r"insert\s+into|create\s+table|partition\s+by|over\s*\(|inner\s+join|"
    r"left\s+join|on\b|and\b|as\b)\b",
    re.I,
)
# Parentheses are deliberately NOT a SQL signal. These notes head real sections
# with them - "SQL (STRUCTURAL QUERY LANGAUGE)", "PRIMARY KEY (UNIQUE+NOT NULL)",
# "1. SCD (Slowly Changing Dimension)" - and treating them as SQL rejected those
# headings, silently merging their sections into whatever came before. Actual SQL
# in this document always carries =, ;, * or two or more SQL keywords.
_SQL_SYNTAX = re.compile(r"[=;*]|[<>!]=|\|\|")

# Words that mark a line as a sentence rather than a label.
_SENTENCE_WORDS = {
    "is", "are", "was", "were", "be", "will", "can", "always", "never",
    "used", "uses", "means", "has", "have", "does", "do", "it", "we", "the",
    "this", "that", "and", "or", "of", "in", "on", "for", "with", "to",
}

# Sub-item labels: "a) Inner join ...", "1. SCD Type 1", "2) Conformed ..."
_SUBITEM = re.compile(r"^\(?([a-z]|\d{1,2})[\.\)]\s+\S")

# Heading emitted by PDFLoader for a rendered comparison table.
_VS_HEADING = re.compile(r"^[^\n]{2,46}\s+vs\s+[^\n]{2,46}$")

# Lines that look like headings but are really continuations of the topic that
# precedes them. "Join" followed by "Types of joins:" is ONE section; treating
# the second as a new section orphans the definition sentence from the list.
_CONTINUATION_HEADING = re.compile(
    r"^\s*(types?\s+of|advantages?|disadvantages?|dis-advantages?|benefits?|"
    r"uses?\s+of|purpose|syntax|examples?|difference\s+between|kinds?\s+of|"
    r"why|when|how)\b",
    re.I,
)

# Generic data labels that must never be promoted to headings.
_GENERIC_LABELS = {
    "source table", "target table", "status", "remark", "name", "output",
    "description", "sql query", "expected result", "actual result", "input",
    "test case no", "scenario", "options", "example", "syntax", "answer",
    "loads", "flag", "level", "id", "city", "version", "table_a", "table_b",
    "input1", "input2", "output1", "output2", "emp", "dept", "note", "notes",
    "query", "result", "solution", "types", "definition", "point", "points",
}


def _is_sqlish(line: str) -> bool:
    """True if the line is SQL or SQL-bearing prose rather than a label."""
    if _SQL_SYNTAX.search(line):
        return True
    # Two or more SQL keywords means it is a statement, not a heading.
    return len(_SQL_MARKERS.findall(line)) >= 2


def _anchor_is_standalone(line: str) -> bool:
    """True if a topic anchor DOMINATES the line, i.e. the line is a label like
    "Join" or "Left Outer Join:" rather than prose that happens to contain the
    word. Requires the anchor to cover a large share of the line's letters."""
    upper = line.upper().rstrip(": ").strip()
    letters = len(re.sub(r"[^A-Z0-9]", "", upper))
    if not letters:
        return False
    for anchor in TOPIC_ANCHORS:
        if anchor not in upper:
            continue
        a_letters = len(re.sub(r"[^A-Z0-9]", "", anchor))
        if a_letters / letters >= 0.45:
            return True
        # A line that merely STARTS with an anchor counts only when what
        # follows is a short qualifier ("Agile Methodology", "Join Types") and
        # not a sentence. Without this, comparison-table cells such as
        # "Primary Key is always alphanumeric" register as headings.
        if upper.startswith(anchor):
            rest = line[len(anchor):].strip(" :-").split()
            if len(rest) <= 2 and not any(
                w.lower() in _SENTENCE_WORDS for w in rest
            ):
                return True
    return False


def is_tabular_page(lines) -> bool:
    """True if a page is a flattened two-column comparison table rather than
    prose with headings.

    Several pages of these notes are "X vs Y" tables. PyMuPDF flattens them into
    alternating single lines, so the left and right column values interleave and
    ordinary heading heuristics fire on almost every other line (`OLTP data
    base`, `data warehouse`, `fact table`). On such pages only true topic
    anchors are allowed to open a section; the generic all-caps and
    trailing-colon rules are suppressed."""
    if len(lines) < 8:
        return False
    candidates = sum(1 for ln in lines if _looks_like_label(ln))
    if candidates < 5:
        return False
    total = sum(len(ln) for ln in lines)
    # Many candidate labels with very little text between them means tabular.
    return total / max(candidates, 1) < 110


def _looks_like_label(line: str) -> bool:
    """Loose label test used only for tabular-page detection."""
    line = line.strip()
    if not (3 <= len(line) <= 62) or line.startswith("•") or _is_sqlish(line):
        return False
    return line.isupper() or line.endswith(":") or line.istitle()


def is_section_heading(line: str, tabular: bool = False) -> bool:
    """Strict level-1 heading test: a standalone topic label.

    On a tabular page only anchor-based headings qualify (see is_tabular_page).
    """
    line = line.strip()
    if not (3 <= len(line) <= 62):
        return False
    if line.startswith("•"):
        return False
    # "X vs Y" is emitted by PDFLoader for a comparison table it recovered, so
    # it is trusted and checked BEFORE the SQL-ish filter. Several such headings
    # contain parentheses ("NORMAL DATABASE (OLTP) vs DATA WAREHOUSE (OLAP)"),
    # which the filter would otherwise reject, silently folding the table into
    # whichever section happened to be open.
    if _VS_HEADING.match(line):
        return True
    if _is_sqlish(line):
        return False
    if _SUBITEM.match(line):
        return False  # that is a level-2 sub-item, handled separately
    if _CONTINUATION_HEADING.match(line):
        return False  # belongs to the topic above it
    label = line.rstrip(":").strip().lower()
    if label in _GENERIC_LABELS:
        return False
    if _anchor_is_standalone(line):
        return True
    if tabular:
        # Only anchors open sections inside a flattened comparison table.
        return False
    # A lone word that is not a known anchor is almost always data, not a
    # heading: this document contains DESC output whose column names
    # (REGNUMBER, STUDENTNAME, PHONENUMBER, BRANCH) are short all-caps lines.
    if len(label.split()) < 2:
        return False
    # Reject table data rows. The notes embed SCOTT.EMP dumps whose rows
    # ("MANAGER 7839 02-APR-81", "MGR HIREDATE") otherwise look like headings.
    if re.search(r"\d{2}-[a-z]{3}-\d{2}|\b\d{4}\b", label):
        return False
    if sum(c.isdigit() for c in label) > len(label) * 0.15:
        return False
    if len([w for w in re.findall(r"[a-z]{3,}", label)]) < 2:
        return False
    if line.isupper() and len(line) <= 50:
        return True
    # Title-case standalone label ending in a colon, e.g. "Defect Life Cycle:"
    if line.endswith(":") and len(label) >= 4 and not _is_sqlish(line):
        if len(label.split()) <= 6:
            return True
    return False


def is_subitem_heading(line: str) -> bool:
    """Level-2 heading, e.g. "d) Left Outer Join:" or "3. SCD Type 3"."""
    line = line.strip()
    if not _SUBITEM.match(line) or len(line) > 90:
        return False
    if _is_sqlish(line):
        return False
    # An enumeration entry inside a "Types of joins:" list is short and has no
    # trailing prose; treat both as sub-items, the section builder dedupes.
    return True


def _covered_topics(text: str, exclude: str = "", limit: int = 6) -> list:
    """Topic labels that appear as their own line inside `text`.

    Used to enrich a chunk's contextual header so that a concept defined inside
    a flattened comparison table is retrievable by its own name even when the
    chunk's heading belongs to a neighbouring table cell."""
    found, seen = [], {exclude.strip().lower()}
    for raw in text.split("\n"):
        line = raw.strip().rstrip(":").strip()
        if not (2 <= len(line) <= 40) or _is_sqlish(line):
            continue
        if not _anchor_is_standalone(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(line)
        if len(found) >= limit:
            break
    return found


def derive_topic(heading: str, default: str = "General") -> str:
    """Map a heading line to a short, stable topic label."""
    h = heading.upper()
    mapping = [
        ("SCD TYPE 1", "SCD Type 1"), ("SCD TYPE 2", "SCD Type 2"),
        ("SCD TYPE 3", "SCD Type 3"), ("SLOWLY CHANGING", "SCD"),
        ("FACT TABLE", "Fact Table"), ("DIMENSION TABLE", "Dimension Table"),
        ("NORMALIZATION", "Normalization"), ("NORMAL FORM", "Normal Forms"),
        ("BCNF", "Normalization"), ("1NF", "Normalization"),
        ("2NF", "Normalization"), ("3NF", "Normalization"),
        ("SURROGATE KEY", "Database Keys"), ("PRIMARY KEY", "Database Keys"),
        ("FOREIGN KEY", "Database Keys"), ("COMPOSITE KEY", "Database Keys"),
        ("STAR SCHEMA", "Star/Snowflake Schema"),
        ("SNOWFLAKE SCHEMA", "Star/Snowflake Schema"),
        ("DATA WAREHOUSE", "Data Warehouse"), ("DATA MART", "Data Warehouse"),
        ("OLTP", "OLTP vs OLAP"), ("OLAP", "OLTP vs OLAP"),
        ("JOIN", "SQL Joins"), ("UNION", "Set Operators"),
        ("INTERSECT", "Set Operators"), ("MINUS", "Set Operators"),
        ("TRUNCATE", "SQL Commands"), ("DELETE", "SQL Commands"),
        ("DROP", "SQL Commands"), ("DDL", "SQL Commands"),
        ("DML", "SQL Commands"), ("CONSTRAINT", "Constraints"),
        ("ROW_NUMBER", "Analytical Functions"), ("DENSE_RANK", "Analytical Functions"),
        ("RANK", "Analytical Functions"), ("LEAD", "Analytical Functions"),
        ("LAG", "Analytical Functions"), ("WINDOW FUNCTION", "Analytical Functions"),
        ("ANALYTICAL FUNCTION", "Analytical Functions"),
        ("NVL", "Null Handling"), ("COALESCE", "Null Handling"),
        ("AGILE", "Agile"), ("SCRUM", "Agile"), ("SPRINT", "Agile"),
        ("DEFECT", "Defect Life Cycle"), ("BUG", "Defect Life Cycle"),
        ("SEVERITY", "Severity vs Priority"), ("PRIORITY", "Severity vs Priority"),
        ("SDLC", "SDLC/STLC"), ("STLC", "SDLC/STLC"),
        ("TEST CASE", "Testing"), ("TESTING", "Testing"),
        ("UNIX", "Unix Commands"), ("SELF INTRODUCTION", "Self Introduction"),
        ("ARCHITECTURE", "Project Architecture"), ("ETL", "ETL Concepts"),
    ]
    for key, topic in mapping:
        if key in h:
            return topic
    clean = re.sub(r"^[\d\.\)a-z]{0,4}[\.\)]?\s*", "", heading.strip()).strip(": ").strip()
    if clean and len(clean) <= 40:
        return clean.title()
    return default


def reflow_lines(lines, wrap_threshold=84):
    """Rejoin PDF hard-wrapped lines into whole sentences.

    PyMuPDF wraps this document near ~99 chars, so a long line whose successor
    begins lowercase is a continuation, not a new point. Leaving them split
    degrades both embedding quality and the readability of extracted answers.
    """
    out = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if out:
            prev = out[-1]
            starts_new = (
                line.startswith("•")
                or _SUBITEM.match(line)
                or line[:1].isupper()
                or line[:1].isdigit()
            )
            # Continuation: previous line was near the wrap width, did not end
            # a sentence, and this line does not look like a fresh point.
            if (
                len(prev) >= wrap_threshold
                and not prev.endswith((".", ":", ";", "?"))
                and not starts_new
            ):
                out[-1] = prev + " " + line
                continue
        out.append(line)
    return out


class HierarchicalChunker:
    """Produces (parents, children). Children are indexed; parents are fed to
    the LLM."""

    def __init__(self, max_parent_chars=2600, target_child_chars=420,
                 max_child_chars=700, min_child_chars=80,
                 min_parent_chars=180, detect_tabular=False):
        # Tabular-page detection predates column-aware extraction in
        # pdf_loader. Now that find_tables() recovers the real columns and the
        # interleaved copy is stripped, the heuristic mostly misfires: it read
        # page 11 (a list of SQL commands and data types) as tabular, which
        # suppressed the all-caps heading rule and swallowed the
        # "SQL (STRUCTURAL QUERY LANGAUGE)" section into its neighbour.
        # Off by default; kept so it can be re-enabled and measured.
        self.detect_tabular = detect_tabular
        self.max_parent_chars = max_parent_chars
        self.target_child_chars = target_child_chars
        self.max_child_chars = max_child_chars
        self.min_child_chars = min_child_chars
        self.min_parent_chars = min_parent_chars

    # -- section splitting ------------------------------------------------
    def _split_sections(self, pages):
        """Group all lines into page-spanning sections keyed by level-1
        heading. Returns list of {topic, heading, start_page, lines}."""
        sections = []
        current = None

        for p in pages:
            page_no = p["page"]
            page_lines = reflow_lines(p["text"].split("\n"))
            tabular = self.detect_tabular and is_tabular_page(page_lines)
            for line in page_lines:
                if is_section_heading(line, tabular=tabular):
                    if current and current["lines"]:
                        sections.append(current)
                    current = {
                        "heading": line.rstrip(":").strip(),
                        "topic": derive_topic(line),
                        "start_page": page_no,
                        "end_page": page_no,
                        "lines": [],
                    }
                    continue
                if current is None:
                    current = {
                        "heading": "Notes",
                        "topic": "General",
                        "start_page": page_no,
                        "end_page": page_no,
                        "lines": [],
                    }
                current["lines"].append({"page": page_no, "text": line})
                current["end_page"] = page_no

        if current and current["lines"]:
            sections.append(current)
        return self._merge_thin_sections(sections)

    def _merge_thin_sections(self, sections):
        """Fold a section that is too small to stand alone into its
        predecessor. A heading followed by one or two lines is a sub-part of the
        preceding topic, not a topic of its own, and indexing it separately is
        what previously produced 200-char orphan chunks."""
        merged = []
        for s in sections:
            size = sum(len(l["text"]) for l in s["lines"])
            # A short section whose heading is a real topic anchor is a topic in
            # its own right, so it is kept standalone. Page 57 of these notes is
            # a run of two-column comparison tables (JOIN vs UNION, PK vs
            # UNIQUE, SURROGATE KEY ...); merging those by size alone produced
            # one grab-bag section titled "JOIN" that answered none of them well.
            if _anchor_is_standalone(s["heading"]) and size >= 60:
                merged.append(s)
                continue
            if merged and size < self.min_parent_chars:
                prev = merged[-1]
                # Keep the dropped heading as a line so its wording stays
                # searchable inside the parent.
                prev["lines"].append({"page": s["start_page"], "text": s["heading"]})
                prev["lines"].extend(s["lines"])
                prev["end_page"] = max(prev["end_page"], s["end_page"])
            else:
                merged.append(s)
        return merged

    # -- child splitting -------------------------------------------------
    def _split_children(self, section):
        """Split one section's lines into small retrievable units, breaking at
        sub-item labels and at the target size."""
        groups = []
        buf = []
        # `label` persists across size-based flushes so that the second half of
        # a long sub-item still reports which sub-item it came from.
        label = None

        def flush():
            nonlocal buf
            if not buf:
                return
            text = "\n".join(l["text"] for l in buf).strip()
            if text:
                groups.append({
                    "label": label,
                    "page": buf[0]["page"],
                    "text": text,
                })
            buf = []

        for line in section["lines"]:
            text = line["text"]
            is_label = is_subitem_heading(text) or (
                # A continuation label only counts when it is a short label,
                # not "Example: SELECT A.X, B.X FROM ...;" which is a whole
                # SQL statement that happens to start with "Example:".
                _CONTINUATION_HEADING.match(text)
                and len(text) <= 60
                and not _is_sqlish(text)
            )
            if is_label:
                flush()
                label = text.rstrip(":").strip()
            buf.append(line)
            if len("\n".join(l["text"] for l in buf)) >= self.target_child_chars:
                flush()
        flush()

        # Merge undersized neighbours so we do not re-create tiny fragments.
        merged = []
        for g in groups:
            if merged and len(merged[-1]["text"]) < self.min_child_chars:
                merged[-1]["text"] += "\n" + g["text"]
                merged[-1]["label"] = merged[-1]["label"] or g["label"]
            else:
                merged.append(g)
        return merged

    def _split_parent_body(self, body):
        """Split an oversized section into parts on line boundaries."""
        if len(body) <= self.max_parent_chars:
            return [body]
        parts, buf = [], []
        for line in body.split("\n"):
            if buf and len("\n".join(buf)) + len(line) > self.max_parent_chars:
                parts.append("\n".join(buf))
                buf = []
            buf.append(line)
        if buf:
            parts.append("\n".join(buf))
        return parts

    # -- public ----------------------------------------------------------
    def chunk_pages(self, pages):
        parents, children = [], []

        for section in self._split_sections(pages):
            # The heading line is re-attached to the body so the section text
            # is self-contained and no source wording is dropped.
            body = "\n".join(
                [section["heading"]] + [l["text"] for l in section["lines"]]
            ).strip()
            if not body:
                continue

            page_label = (
                f"page {section['start_page']}"
                if section["start_page"] == section["end_page"]
                else f"pages {section['start_page']}-{section['end_page']}"
            )

            # A long section (Unix Commands runs pages 59-61, NVL2 21-24) is
            # split into sequential parts rather than truncated, so no note
            # text is ever dropped from the knowledge base.
            parts = self._split_parent_body(body)
            parent_ids = []
            for n, part in enumerate(parts):
                pid = len(parents)
                parent_ids.append(pid)
                suffix = f" (part {n + 1}/{len(parts)})" if len(parts) > 1 else ""
                parents.append({
                    "parent_id": pid,
                    "topic": section["topic"],
                    "heading": section["heading"] + suffix,
                    "page": section["start_page"],
                    "end_page": section["end_page"],
                    "page_label": page_label,
                    "content": part,
                })

            # Children map to the parent part whose text contains them.
            def owning_parent(text):
                for pid, part in zip(parent_ids, parts):
                    if text[:60] in part:
                        return pid
                return parent_ids[0]

            for g in self._split_children(section):
                # Contextual header: makes the fragment self-describing for
                # both BM25 and the embedding model (Anthropic-style
                # contextual retrieval, done deterministically from the PDF's
                # own structure rather than with an LLM pass).
                crumb = section["heading"]
                if g["label"] and g["label"].lower() != crumb.lower():
                    crumb = f"{crumb} > {g['label']}"
                header = f"Section: {crumb} ({section['topic']}, page {g['page']})"

                # Some pages are two-column comparison tables that PyMuPDF
                # flattens, so a chunk's own heading can be an unrelated table
                # cell while its body defines several concepts. Naming the topic
                # labels found inside the body makes the chunk findable by them
                # instead of only by its heading.
                covers = _covered_topics(g["text"], exclude=crumb)
                if covers:
                    header += f" | covers: {', '.join(covers)}"
                children.append({
                    "parent_id": owning_parent(g["text"]),
                    "topic": section["topic"],
                    "heading": section["heading"],
                    "label": g["label"],
                    "page": g["page"],
                    "header": header,
                    "content": g["text"][: self.max_child_chars],
                    # what actually gets embedded / BM25-indexed
                    "index_text": f"{header}\n{g['text'][: self.max_child_chars]}",
                })

        return parents, children


# Backwards-compatible alias: build_index.py used SemanticChunker before.
SemanticChunker = HierarchicalChunker
