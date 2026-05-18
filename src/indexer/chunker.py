import uuid
from dataclasses import dataclass, field
from src.indexer.detector import detect_structure

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
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[Chunk]:
    units = detect_structure(text)
    chunks: list[Chunk] = []
    current_section_path: list[str] = []
    current_section_title: str = ""
    current_parent_id: str | None = None
    current_root_id: str | None = None
    prose_buffer: list[str] = []

    def flush_prose():
        nonlocal prose_buffer, current_parent_id
        if not prose_buffer:
            return
        full_text = " ".join(prose_buffer)
        # Split into leaf chunks if too long
        words = full_text.split()
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
                page_start=page_start,
                page_end=page_end,
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
                current_root_id = None  # reset before creating this chunk — depth-1 chunks have no root parent
            current_section_path = current_section_path[: depth - 1] + [unit.section_number]
            current_section_title = unit.text
            parent = Chunk(
                chunk_id=_make_id(),
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                text=unit.text,
                chunk_type="section",
                is_leaf=False,
                root_chunk_id=current_root_id,  # now None for all depth-1 headings
                section_path=list(current_section_path),
                section_title=current_section_title,
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(parent)
            current_parent_id = parent.chunk_id
            if depth == 1:
                current_root_id = parent.chunk_id  # descendants of this section will use this as root

        elif unit.kind == "empfehlung":
            flush_prose()
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
                page_start=page_start,
                page_end=page_end,
                chunk_index_in_parent=None,
            )
            chunks.append(leaf)

        elif unit.kind == "prose":
            prose_buffer.append(unit.text)

        elif unit.kind == "bibliography_entry":
            flush_prose()  # Don't include bib entries as retrieval chunks

    flush_prose()
    return chunks
