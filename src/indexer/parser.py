import re
from pathlib import Path
from typing import TypedDict
import fitz  # pymupdf


class PageDict(TypedDict):
    page_number: int
    text: str


def extract_pages(pdf_path: Path) -> list[PageDict]:
    """Extract text page by page from a PDF. Returns list of {page_number, text}."""
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({"page_number": i + 1, "text": text})
    return pages


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text:
    - Repair German hyphenation across line breaks
    - Merge broken paragraph lines
    - Strip trailing whitespace per line
    - Preserve section numbers and Empfehlung labels
    """
    # Only repair hyphenation when both sides are lowercase German letters — this is the
    # soft-hyphenation pattern for compound words. Uppercase right-hand side means a new
    # capitalised word or proper noun; numeric left-hand side means a range, not a compound.
    text = re.sub(r"([a-zäöüß])-\n([a-zäöüß])", r"\1\2", text)

    lines = text.splitlines()
    merged: list[str] = []
    i = 0
    # Merge continuation lines pair-by-pair. German PDF extraction frequently wraps
    # prose at ~80 chars; we join the current line with its successor when neither
    # looks like a structural boundary. Running prose across 3+ lines is handled
    # implicitly: after merging lines i and i+1, the cursor advances to i+2 and
    # the same check applies there.
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


def _is_heading(stripped_line: str) -> bool:
    # Expects a pre-stripped string. Numbered section headings (1, 1.2, 3.4.1)
    if re.match(r"^\d+(\.\d+)*\s+\S", stripped_line):
        return True
    # Recommendation labels: Empfehlung X.Y or Empfehlungsgrad / Evidenzlevel markers
    if re.match(r"^(Empfehlung|Empfehlungsgrad|Evidenzlevel)\b", stripped_line, re.IGNORECASE):
        return True
    return False


def _ends_sentence(line: str) -> bool:
    return line.rstrip().endswith((".", ":", "!", "?"))
