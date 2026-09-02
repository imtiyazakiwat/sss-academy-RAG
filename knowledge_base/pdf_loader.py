"""
PDF text extraction using PyMuPDF.
Extracts per-page and per-block text while preserving page numbers
and structural information (headings, paragraphs, bullets).
"""

import os
import re
import pymupdf


class PDFLoader:
    def __init__(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        self.pdf_path = pdf_path

    def extract_pages(self, tables=True):
        """Extract text page by page, with two-column comparison tables
        rendered as explicitly labelled lines.

        Why this matters: much of these notes is "X vs Y" comparison tables
        (DELETE vs TRUNCATE, PRIMARY KEY vs SURROGATE KEY, OLTP vs OLAP). Plain
        get_text("text") flattens them into interleaved lines, so which column a
        statement belongs to is lost:

            DELETE
            TRUNCATE
            We delete the specific row or whole data from the table.
            We delete all the data from the table at a single shot.

        A reader cannot tell which line describes which command, and neither can
        the model, which produced answers with the two sides swapped. Detecting
        the table and labelling every line removes the ambiguity."""
        doc = pymupdf.open(self.pdf_path)
        pages = []
        for idx, page in enumerate(doc):
            text = page.get_text("text") or ""
            rendered = []
            if tables:
                try:
                    found = page.find_tables()
                except Exception:
                    found = None
                for table in (found.tables if found else []):
                    block, cells = self._render_table(table)
                    if block:
                        rendered.append(block)
                        # Drop the raw interleaved copy so the same content is
                        # not indexed twice in its ambiguous form.
                        text = _strip_cells(text, cells)

            combined = "\n\n".join([text.strip()] + rendered).strip()
            if combined:
                pages.append({"page": idx + 1, "text": combined})
        doc.close()
        return pages

    @staticmethod
    def _render_table(table):
        """Render a 2-column comparison table as labelled lines.
        Returns (text_block, list_of_cell_strings)."""
        try:
            rows = table.extract()
        except Exception:
            return "", []
        rows = [[(c or "").strip() for c in r] for r in rows]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2 or len(rows[0]) != 2:
            return "", []

        left_label, right_label = rows[0][0], rows[0][1]
        if not (left_label and right_label):
            return "", []
        # Headers must look like labels, not sentences.
        if len(left_label) > 46 or len(right_label) > 46:
            return "", []

        # Not every two-column table is a comparison. Page 44 holds a
        # severity-level example table (Urgent / Without steering, Very High /
        # Break not working) whose first row is data rather than headers;
        # rendering it produced a nonsense "Urgent vs Without steering" section.
        # A real comparison here always has at least one descriptive cell, while
        # example and lookup tables are uniformly short.
        body = [c for r in rows[1:] for c in r if c]
        if not body or max(len(c) for c in body) < 32:
            return "", []

        cells = [c for r in rows for c in r if c]
        lines = [f"{left_label} vs {right_label}"]
        for left, right in ((r[0], r[1]) for r in rows[1:]):
            if left:
                lines.append(f"• {left_label}: {_clean_cell(left)}")
            if right:
                lines.append(f"• {right_label}: {_clean_cell(right)}")
        return "\n".join(lines), cells


def _clean_cell(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.lstrip("•").strip()


def _strip_cells(page_text: str, cells) -> str:
    """Remove lines already captured by a rendered table."""
    if not cells:
        return page_text
    blob = " ".join(re.sub(r"\s+", " ", c) for c in cells).lower()
    kept = []
    for line in page_text.split("\n"):
        probe = re.sub(r"\s+", " ", line).strip().lstrip("•").strip().lower()
        if len(probe) >= 4 and probe in blob:
            continue
        kept.append(line)
    return "\n".join(kept)

    def extract_blocks(self):
        """Extract text blocks with page and block indices.
        Blocks preserve layout better than raw page text for chunking."""
        doc = pymupdf.open(self.pdf_path)
        blocks = []
        for idx, page in enumerate(doc):
            raw_blocks = page.get_text("blocks")
            for b in raw_blocks:
                x0, y0, x1, y1, text, bno, btype = b
                if btype != 0:  # only text blocks
                    continue
                cleaned = text.strip()
                if cleaned:
                    blocks.append({
                        "page": idx + 1,
                        "block": bno,
                        "text": cleaned,
                    })
        doc.close()
        return blocks

    def extract_plain(self):
        """Return full document text (for fallback / debugging)."""
        pages = self.extract_pages()
        return "\n\n".join(p["text"] for p in pages)


def clean_text(text: str) -> str:
    """Normalize extracted text: collapse whitespace, fix artifacts."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but keep newlines)
    lines = [re.sub(r"[ \t]+", " ", ln) for ln in text.split("\n")]
    # Remove standalone bullet artifacts like lines that are just "•"
    cleaned = "\n".join(
        ln for ln in lines
        if ln.strip() not in ("•", "-", ".", ",")
    )
    # Remove repeated "SSS ACADEMY" headers and isolated page numbers
    cleaned = re.sub(r"^\s*SSS\s+ACADEMY\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d{1,3}\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()
