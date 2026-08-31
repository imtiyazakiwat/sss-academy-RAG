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

    def extract_pages(self):
        """Extract text page by page with page numbers."""
        doc = pymupdf.open(self.pdf_path)
        pages = []
        for idx, page in enumerate(doc):
            text = page.get_text("text")
            if text and text.strip():
                pages.append({
                    "page": idx + 1,
                    "text": text.strip(),
                })
        doc.close()
        return pages

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
