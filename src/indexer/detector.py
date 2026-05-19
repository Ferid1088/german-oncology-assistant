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
    kind: str                          # heading | empfehlung | bibliography_entry | prose
    text: str
    section_number: str = ""
    recommendation_id: str = ""
    recommendation_grade: str = ""
    evidence_level: str = ""
    reference_id: str = ""
    line_start: int = 0


def detect_structure(text: str) -> list[StructuralUnit]:
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
            while j < len(lines) and lines[j].strip():
                bl = lines[j].strip()
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
    return bool(re.match(r"^\[\d+\]\s+\w+.+\d{4}", line))
