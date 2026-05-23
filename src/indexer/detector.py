"""Structural detector for German S3 oncology guideline text.

Scans cleaned plaintext line by line and emits a sequence of ``StructuralUnit``
objects, each tagged with one of four kinds:
- ``heading`` — numbered section header (e.g. "4.7.3 Adjuvante Chemotherapie").
- ``empfehlung`` — a recommendation block including grade and evidence level.
- ``bibliography_entry`` — a bracketed reference (e.g. "[42] Author …").
- ``prose`` — any other contiguous non-empty text block.

The parser is a simple one-pass state machine; no LLM is involved.
"""

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3})+)\.?\s+[A-ZÄÖÜ]")
EMPFEHLUNG_RE = re.compile(
    r"^(\d+\.\d+(?:\.\d+)?)\.?\s+(Evidenzbasierte|Konsensbasierte)\s+(Empfehlung|Statement)",
    re.IGNORECASE,
)
GRADE_RE = re.compile(r"Empfehlungsgrad\s*:?\s*([A-Z0])\b", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"(?:Evidenzlevel|Level of Evidence)\s*:?\s*(\S+)", re.IGNORECASE)
BIB_RE = re.compile(r"^\[(\d+)\]\s*\S")


@dataclass
class StructuralUnit:
    """A single classified text unit emitted by ``detect_structure``.

    Attributes:
        kind: Classification: "heading" | "empfehlung" | "bibliography_entry" | "prose".
        text: The raw text content of the unit (multi-line for empfehlung blocks).
        section_number: Dotted section number for headings (e.g. "4.7.3"); empty otherwise.
        recommendation_id: Numeric id for empfehlung blocks (e.g. "4.7"); empty otherwise.
        recommendation_grade: Extracted grade character "A", "B", or "0"; empty otherwise.
        evidence_level: Extracted evidence level string (e.g. "1a"); empty otherwise.
        reference_id: Numeric bibliography id string; empty unless kind == "bibliography_entry".
        line_start: Zero-based line index in the joined document text (used for page mapping).
    """

    kind: str                          # heading | empfehlung | bibliography_entry | prose
    text: str
    section_number: str = ""
    recommendation_id: str = ""
    recommendation_grade: str = ""
    evidence_level: str = ""
    reference_id: str = ""
    line_start: int = 0


def detect_structure(text: str) -> list[StructuralUnit]:
    """Classify every line of cleaned guideline text into ``StructuralUnit`` objects.

    Processes the text in a single forward pass.  Priority order for each line:
    1. Bibliography entry (``[N] …``) — once bibliography mode is active, all matching
       lines are classified as bibliography entries.
    2. Empfehlung block — greedily consumes continuation lines until a structural
       boundary or blank line (after the header) is reached.
    3. Heading — single-line classification.
    4. Prose — accumulates until the next blank line or structural boundary.

    Args:
        text: Pre-cleaned, pre-normalized document text.

    Returns:
        Ordered list of ``StructuralUnit`` objects covering every non-empty line.
    """
    lines = text.splitlines()
    units: list[StructuralUnit] = []
    i = 0
    in_bibliography = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Bibliography section starts when we see consecutive [N] lines
        if BIB_RE.match(stripped) and (in_bibliography or _looks_like_bib(stripped)):
            in_bibliography = True
            m = BIB_RE.match(stripped)
            units.append(StructuralUnit(
                kind="bibliography_entry",
                text=stripped,
                reference_id=m.group(1),
                line_start=i,
            ))
            i += 1
            continue

        if EMPFEHLUNG_RE.match(stripped):
            block_lines = [stripped]
            grade = ""
            evidence = ""
            j = i + 1
            while j < len(lines):
                bl = lines[j].strip()
                if not bl:
                    # EK blocks have one blank line between header and body text;
                    # skip it only when we haven't collected any body lines yet.
                    if len(block_lines) == 1:
                        j += 1
                        continue
                    break
                if HEADING_RE.match(bl) or EMPFEHLUNG_RE.match(bl) or BIB_RE.match(bl):
                    break
                gm = GRADE_RE.search(bl)
                em = EVIDENCE_RE.search(bl)
                if gm:
                    grade = gm.group(1).upper()
                if em:
                    evidence = em.group(1)
                block_lines.append(bl)
                j += 1
            rec_id = EMPFEHLUNG_RE.match(stripped).group(1)  # group 1 = numeric id
            units.append(StructuralUnit(
                kind="empfehlung",
                text="\n".join(block_lines),
                recommendation_id=rec_id,
                recommendation_grade=grade,
                evidence_level=evidence,
                line_start=i,
            ))
            i = j
            continue

        if HEADING_RE.match(stripped):
            m = HEADING_RE.match(stripped)
            units.append(StructuralUnit(
                kind="heading",
                text=stripped,
                section_number=m.group(1),
                line_start=i,
            ))
            i += 1
            continue

        # Prose: accumulate until next structural boundary
        prose_lines = [stripped]
        j = i + 1
        while j < len(lines):
            next_line = lines[j].strip()
            if not next_line:
                break
            if HEADING_RE.match(next_line) or EMPFEHLUNG_RE.match(next_line) or BIB_RE.match(next_line):
                break
            prose_lines.append(next_line)
            j += 1
        units.append(StructuralUnit(kind="prose", text=" ".join(prose_lines), line_start=i))
        i = j

    return units


def _looks_like_bib(line: str) -> bool:
    """Heuristic check for a bibliography entry without prior bibliography context.

    Requires a bracketed number followed by at least one word and a 4-digit year,
    to distinguish from inline citations like "[3]" that appear inside prose.

    Args:
        line: A single stripped text line.

    Returns:
        True when the line looks like the start of a bibliography section.
    """
    return bool(re.match(r"^\[\d+\]\s+\w+.+\d{4}", line))
