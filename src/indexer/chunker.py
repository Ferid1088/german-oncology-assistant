import uuid
from dataclasses import dataclass, field
from src.indexer.detector import detect_structure

# (line_start_in_full_text, doc_page_number)
PageBoundary = tuple[int, int]


def _page_at_line(boundaries: list[PageBoundary], line: int) -> int | None:
    """Return the doc page number for a given line in the joined full text."""
    result = None
    for line_start, page_num in boundaries:
        if line_start <= line:
            result = page_num
        else:
            break
    return result

TARGET_TOKENS = 550       # middle of 400-700 range
OVERLAP_TOKENS = 70       # ~12%


@dataclass
class Chunk:
    chunk_id: str
    guideline_id: str
    guideline_version: str
    text: str
    chunk_type: str           # section | recommendation | evidence | rationale | table
    is_leaf: bool
    parent_chunk_id: str | None = None
    root_chunk_id: str | None = None
    section_path: list[str] = field(default_factory=list)
    section_title: str = ""
    recommendation_id: str = ""
    recommendation_grade: str = ""
    evidence_level: str = ""
    page_start: int | None = None
    page_end: int | None = None
    chunk_index_in_parent: int | None = None
    source_filename: str = ""
    is_current: bool = True


def _make_id() -> str:
    return str(uuid.uuid4())


def build_chunks(
    guideline_id: str,
    guideline_version: str,
    text: str,
    page_boundaries: list[PageBoundary] | None = None,
) -> list[Chunk]:
    units = detect_structure(text)
    chunks: list[Chunk] = []
    current_section_path: list[str] = []
    current_section_title: str = ""
    current_parent_id: str | None = None
    current_root_id: str | None = None
    prose_buffer: list[str] = []
    prose_line_start: int = 0  # line_start of the first prose unit in current buffer

    def _page(line: int) -> int | None:
        return _page_at_line(page_boundaries, line) if page_boundaries else None

    MIN_PROSE_WORDS = 25  # drop author-list lines and other one-liner fragments

    def flush_prose():
        nonlocal prose_buffer, current_parent_id, prose_line_start
        if not prose_buffer:
            return
        pg = _page(prose_line_start)
        full_text = " ".join(prose_buffer)
        words = full_text.split()
        if len(words) < MIN_PROSE_WORDS:
            prose_buffer = []
            return
        start = 0
        chunk_index = 0
        while start < len(words):
            end = start
            token_count = 0
            while end < len(words) and token_count < TARGET_TOKENS:
                token_count = int((end - start + 1) * 1.3)
                end += 1
            chunk_text = " ".join(words[start:end])
            leaf = Chunk(
                chunk_id=_make_id(),
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                text=chunk_text,
                chunk_type="section",
                is_leaf=True,
                parent_chunk_id=current_parent_id,
                root_chunk_id=current_root_id,
                section_path=list(current_section_path),
                section_title=current_section_title,
                page_start=pg,
                page_end=pg,
                chunk_index_in_parent=chunk_index,
            )
            chunks.append(leaf)
            chunk_index += 1
            start = end - int(OVERLAP_TOKENS / 1.3) if end < len(words) else end
        prose_buffer = []

    for unit in units:
        if unit.kind == "heading":
            flush_prose()
            depth = unit.section_number.count(".") + 1
            if depth == 1:
                current_root_id = None
            # Only keep ancestors from current_path that are true numeric ancestors of
            # section_number (i.e. section_number starts with "ancestor.").
            # Without this check, a stale deep path like ["5.4.1"] would be incorrectly
            # prepended to an unrelated sibling section like "3.1".
            ancestors = [
                s for s in current_section_path[: depth - 1]
                if unit.section_number.startswith(s + ".")
            ]
            current_section_path = ancestors + [unit.section_number]
            current_section_title = unit.text
            pg = _page(unit.line_start)
            parent = Chunk(
                chunk_id=_make_id(),
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                text=unit.text,
                chunk_type="section",
                is_leaf=False,
                root_chunk_id=current_root_id,
                section_path=list(current_section_path),
                section_title=current_section_title,
                page_start=pg,
                page_end=pg,
            )
            chunks.append(parent)
            current_parent_id = parent.chunk_id
            if depth == 1:
                current_root_id = parent.chunk_id

        elif unit.kind == "empfehlung":
            flush_prose()
            pg = _page(unit.line_start)
            leaf = Chunk(
                chunk_id=_make_id(),
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                text=unit.text,
                chunk_type="recommendation",
                is_leaf=True,
                parent_chunk_id=current_parent_id,
                root_chunk_id=current_root_id,
                section_path=list(current_section_path),
                section_title=current_section_title,
                recommendation_id=unit.recommendation_id,
                recommendation_grade=unit.recommendation_grade,
                evidence_level=unit.evidence_level,
                page_start=pg,
                page_end=pg,
                chunk_index_in_parent=None,
            )
            chunks.append(leaf)

        elif unit.kind == "prose":
            if not prose_buffer:
                prose_line_start = unit.line_start
            prose_buffer.append(unit.text)

        elif unit.kind == "bibliography_entry":
            flush_prose()

    flush_prose()
    return chunks
