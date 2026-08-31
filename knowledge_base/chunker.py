"""
Semantic chunker.

Splits the PDF text into topic-aligned chunks. Instead of fixed character
sizes, it splits on structural boundaries:
  - Heading-like lines (short lines, title case / capitalized keywords)
  - Numbered sections ("1.", "2.", "3.")
  - Known ETL concept anchors (SCD, Fact Table, Normal Form, etc.)

Each chunk stores metadata: topic, page, and content.
"""

import re
from collections import defaultdict

# Strong topic anchors - lines that indicate a NEW topic boundary
TOPIC_ANCHORS = [
    "SCD TYPE", "SCD TYPE 1", "SCD TYPE 2", "SCD TYPE 3",
    "FACT TABLE", "DIMENSION TABLE", "SLOWLY CHANGING",
    "NORMALIZATION", "NORMAL FORM", "1NF", "2NF", "3NF", "BCNF",
    "PRIMARY KEY", "SURROGATE KEY", "FOREIGN KEY",
    "STAR SCHEMA", "SNOWFLAKE SCHEMA", "DATA WAREHOUSE",
    "TRUNCATE", "DELETE", "DROP",
    "JOIN", "INNER JOIN", "LEFT OUTER JOIN", "CROSS JOIN",
    "UNION", "UNION ALL", "INTERSECT", "MINUS",
    "ROW_NUMBER", "RANK", "DENSE_RANK", "LEAD", "LAG",
    "ANALYTICAL", "WINDOW",
    "AGILE", "SCRUM", "SPRINT", "EPIC", "USER STORY",
    "TEST CASE", "TESTING", "SMOKE TESTING", "REGRESSION",
    "DEFECT", "BUG", "SDLC", "STLC",
    "DDL", "DML", "DCL", "TCL",
    "NVL", "NVL2", "NULLIF", "COALESCE",
    "CASE", "DECODE",
    "VIEW", "MATERIALIZED VIEW",
    "UNIX",
    "DUPLICATE", "2ND HIGHEST", "NTH HIGHEST",
    "SELF INTRODUCTION", "ROLES", "ARCHITECTURE",
]


def looks_like_heading(line: str) -> bool:
    """Heuristic: a plausible section heading."""
    line = line.strip()
    if not line:
        return False
    # Too long to be a heading
    if len(line) > 80:
        return False
    # Too short (bullet artifacts)
    if len(line) < 3:
        return False
    upper = line.upper()
    # Known anchor match
    for anchor in TOPIC_ANCHORS:
        if anchor in upper:
            # Avoid matching anchor inside a long run-on sentence
            return True
    # Numbered heading like "1. SCD Type 2" or "1) Introduction"
    if re.match(r"^\d+[\.\)]\s+[A-Z]", line):
        return True
    # Test-case phase labels that define coherent SCD validation blocks
    if re.match(r"^Test Case (for|For)", line) or "Test Case for Initial Load" in line \
       or "Test Case for Incremental Load" in line:
        return True
    # Colon labels only count as headings when STRONG (declarative heading,
    # not a generic data label like "Source Table:" or "Status:" or "Name:")
    if line.endswith(":"):
        generic_labels = {
            "source table", "target table", "status", "remark", "name",
            "description", "sql query", "expected result", "actual result",
            "test case no", "scenario", "options", "example", "syntax",
            "loads", "flag", "case", "level", "id", "city", "version",
        }
        label = line.rstrip(":").strip().lower()
        if label not in generic_labels and len(label) >= 4:
            return True
    # All-caps short line (strong heading)
    if line.isupper() and len(line) <= 50:
        return True
    return False


def derive_topic(heading: str, default: str = "General") -> str:
    """Map a heading line to a short, reusable topic label."""
    # Specific keyword matches first
    h = heading.upper()
    mapping = {
        "SCD TYPE 1": "SCD Type 1",
        "SCD TYPE 2": "SCD Type 2",
        "SCD TYPE 3": "SCD Type 3",
        "SLOWLY CHANGING": "SCD",
        "FACT TABLE": "Fact Table",
        "DIMENSION TABLE": "Dimension Table",
        "NORMALIZATION": "Normalization",
        "NORMAL FORM": "Normal Forms",
        "1NF": "Normalization",
        "2NF": "Normalization",
        "3NF": "Normalization",
        "BCNF": "Normalization",
        "PRIMARY KEY": "Database Keys",
        "SURROGATE KEY": "Database Keys",
        "FOREIGN KEY": "Database Keys",
        "STAR SCHEMA": "Star/Snowflake Schema",
        "SNOWFLAKE SCHEMA": "Star/Snowflake Schema",
        "DATA WAREHOUSE": "Data Warehouse",
        "TRUNCATE": "SQL Commands",
        "DELETE": "SQL Commands",
        "DROP": "SQL Commands",
        "AGILE": "Agile",
        "SCRUM": "Agile",
        "SPRINT": "Agile",
        "TEST CASE": "Testing",
        "DEFECT": "Defect Life Cycle",
        "BUG": "Defect Life Cycle",
        "SDLC": "SDLC/STLC",
        "STLC": "SDLC/STLC",
        "UNIX": "Unix Commands",
        "SELF INTRODUCTION": "Self Introduction",
        "ARCHITECTURE": "Project Architecture",
    }
    for key, topic in mapping.items():
        if key in h:
            return topic
    # Fallback: use the heading itself if short, else "General"
    clean = re.sub(r"^[\d\.\):\s]+", "", heading.strip()).strip()
    if clean and len(clean) <= 40:
        return clean.title()
    return default


class SemanticChunker:
    def __init__(self, min_chunk_chars=200, max_chunk_chars=1500):
        self.min_chars = min_chunk_chars
        self.max_chars = max_chunk_chars

    def chunk_pages(self, pages):
        """pages: list of {page, text}. Returns list of chunk dicts."""
        # 1. Split each page into lines, classify headings vs body
        # 2. Accumulate lines into chunks keyed by (page, current topic)

        chunks = []
        current_chunk_lines = []
        current_topic = "General"
        current_page = None

        def flush():
            nonlocal current_chunk_lines
            if not current_chunk_lines:
                return
            text = "\n".join(current_chunk_lines).strip()
            if text:
                chunks.append({
                    "topic": current_topic,
                    "page": current_page,
                    "content": text[:self.max_chars],
                })
            current_chunk_lines = []

        for p in pages:
            page_no = p["page"]
            lines = p["text"].split("\n")
            # Track if we crossed into a new page -> flush (safety)
            if current_page is not None and page_no != current_page:
                flush()
            current_page = page_no

            for raw_line in lines:
                line = raw_line.rstrip()
                if not line.strip():
                    continue

                if looks_like_heading(line):
                    # If we already have content, flush before starting new topic
                    if current_chunk_lines and len("\n".join(current_chunk_lines).strip()) >= self.min_chars:
                        flush()
                    current_topic = derive_topic(line, current_topic)
                    current_chunk_lines.append(line.strip())
                else:
                    current_chunk_lines.append(line.strip())

                    # Enforce max size: flush greedy long chunks
                    if len("\n".join(current_chunk_lines)) >= self.max_chars:
                        flush()

        flush()
        return chunks
