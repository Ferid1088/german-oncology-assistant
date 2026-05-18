import re
from pathlib import Path
import fitz  # pymupdf


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text page by page from a PDF. Returns list of {page_number, text}."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text:
    - Repair German hyphenation across line breaks
    - Merge broken paragraph lines
    - Strip trailing whitespace per line
    - Preserve section numbers and Empfehlung labels
    """
    # Repair hyphenation: "Behand-\nlung" -> "Behandlung"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = text.splitlines()
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if (
            i + 1 < len(lines)
            and line
            and not _is_heading(line)
            and not _ends_sentence(line)
            and not _is_heading(lines[i + 1].strip())
            and lines[i + 1].strip()
        ):
            # Merge continuation line
            merged.append(line + " " + lines[i + 1].strip())
            i += 2
        else:
            merged.append(line)
            i += 1

    return "\n".join(merged)


def _is_heading(line: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*\s+\S", line.strip()))


def _ends_sentence(line: str) -> bool:
    return line.rstrip().endswith((".", ":", "!", "?"))
