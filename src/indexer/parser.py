"""PDF text extraction and cleaning for German S3 oncology guideline PDFs.

Handles the specific formatting conventions of the Leitlinienprogramm Onkologie:
- Extracts printed page numbers from the two-line copyright header that appears on
  every physical page (``© Leitlinienprogramm Onkologie | … | Juni 2021 \n 238``).
- Strips the copyright block so its page number does not bleed into prose chunks.
- Skips table-of-contents pages detected by dot-leader lines (e.g. "4.3.3.7 … 94").
- Repairs soft German hyphenation across line breaks (e.g. "chemo-\ntherapie").
- Merges continuation lines so the structural detector sees clean paragraphs.
- Normalises split recommendation headers so ``detect_structure`` can match them.
"""

import re
from pathlib import Path
from typing import TypedDict
import fitz  # pymupdf


class PageDict(TypedDict):
    page_number: int       # physical PDF page (1-based)
    doc_page_number: int | None  # printed page number from header, if found
    text: str


# The S3-Leitlinien header spans two lines in fitz output:
#   "© Leitlinienprogramm Onkologie | ... | Juni 2021 "
#   "238"
# Match the copyright line then capture the number on the following line.
_HEADER_PAGE_RE = re.compile(
    r"Leitlinienprogramm\s+Onkologie[^\n]*\n\s*(\d{1,4})\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Strip the full two-line copyright block from each page before cleaning.
# clean_text() would otherwise merge the number into the copyright line,
# producing "... | Juni 2021 70" in every chunk that crosses a page boundary.
_HEADER_STRIP_RE = re.compile(
    r"©[^\n]*Leitlinienprogramm\s+Onkologie[^\n]*\n[ \t]*\d{1,4}[ \t]*\n?",
    re.IGNORECASE,
)

# TOC pages are identified by dot-leader lines: "4.3.3.7 Papillom ........... 94"
_TOC_LEADER_RE = re.compile(r"\.{5,}\s*\d+\s*$")


def _extract_doc_page_number(text: str) -> int | None:
    """Return the printed page number from the S3-Leitlinie header, or None."""
    m = _HEADER_PAGE_RE.search(text[:600])  # header is always near the top
    return int(m.group(1)) if m else None


def _is_toc_page(text: str) -> bool:
    """Return True if the page is a table-of-contents page (dot-leader entries)."""
    toc_lines = sum(1 for l in text.splitlines() if _TOC_LEADER_RE.search(l.strip()))
    return toc_lines >= 5


def extract_pages(pdf_path: Path) -> list[PageDict]:
    """Extract text page by page from a PDF. Returns list of {page_number, doc_page_number, text}."""
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text("text")
            doc_page = _extract_doc_page_number(raw)
            text = _HEADER_STRIP_RE.sub("", raw)
            if _is_toc_page(text):
                text = ""
            pages.append({
                "page_number": i + 1,
                "doc_page_number": doc_page,
                "text": text,
            })
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
        line = lines[i].strip()
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
    """Return True if the line is a structural heading or recommendation marker.

    Used by ``clean_text`` to decide whether to merge the current line with the next.
    Structural lines must never be merged with their successor.

    Args:
        stripped_line: A single stripped line of text.

    Returns:
        True for numbered headings, recommendation lines, and grade/evidence markers.
    """
    # Numbered headings: require at least X.Y (two dotted parts) + uppercase first word.
    # e.g. "4.7.3. Adjuvante Chemotherapie" or "4.116. Evidenzbasierte Empfehlung"
    if re.match(r"^\d{1,3}(?:\.\d{1,3})+\.?\s+[A-ZÄÖÜ]", stripped_line):
        return True
    # Grade, evidence, and recommendation markers (no colon required)
    if re.match(r"^(Empfehlung|Empfehlungsgrad|Evidenzlevel|Level of Evidence)\b", stripped_line, re.IGNORECASE):
        return True
    return False


def _ends_sentence(line: str) -> bool:
    """Return True if the line ends with a sentence-terminating punctuation mark.

    Used by ``clean_text`` to decide whether the next line is a continuation.
    A line ending with '.', ':', '!', or '?' is treated as complete.

    Args:
        line: A single line of text (may have trailing whitespace).

    Returns:
        True when the line should not be merged with its successor.
    """
    return line.rstrip().endswith((".", ":", "!", "?"))


def normalize_recommendations(text: str) -> str:
    """
    German S3 Leitlinien PDFs split recommendation headers across lines:
      3.4.
      Evidenzbasierte Empfehlung
      Empfehlungsgrad
      A
      <prose>

    Merge them into single lines so detect_structure can match them:
      3.4. Evidenzbasierte Empfehlung
      Empfehlungsgrad A
      <prose>
    """
    # Join "N.N.\nEvidenzbasierte/Konsensbasierte Empfehlung" → "N.N. Evidenzbasierte Empfehlung"
    text = re.sub(
        r"^(\d+\.\d+(?:\.\d+)?)\.?\s*\n((?:Evidenzbasierte|Konsensbasierte)\s+(?:Empfehlung|Statement))",
        r"\1. \2",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Join "Empfehlungsgrad\nA" → "Empfehlungsgrad A"
    text = re.sub(
        r"^(Empfehlungsgrad)\s*\n([AB0])\b",
        r"\1 \2",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Join "Level of Evidence\n1a" → "Level of Evidence 1a"
    text = re.sub(
        r"^(Level of Evidence|Evidenzlevel)\s*\n(\S+)",
        r"\1 \2",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return text
