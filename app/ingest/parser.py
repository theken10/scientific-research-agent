import re
from pathlib import Path

import fitz  # pymupdf


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join words broken across lines
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def parse_pdf(path: str | Path):
    """Returns (title, [(page_number, text), ...]). Page numbers start at 1."""
    path = Path(path)
    pages = []
    with fitz.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or ""
        for i, page in enumerate(pdf, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append((i, text))

    if not title.strip():
        # metadata title is often empty for arxiv pdfs, so use the first line of page 1
        first = pages[0][1][:150] if pages else path.stem
        title = first
    return title.strip(), pages
