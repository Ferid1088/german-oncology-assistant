"""Hierarchical chunker for oncology guideline text.

Converts a flat list of ``StructuralUnit`` objects (from ``detector.detect_structure``)
into a tree of ``Chunk`` records with parent/root linkage.  Three chunk types are
produced:
- ``section`` (parent, is_leaf=False) — created for each numbered heading.
- ``section`` (leaf, is_leaf=True) — sliding-window prose slices under a heading.
- ``recommendation`` (leaf, is_leaf=True) — a complete "Empfehlung" block with grade.

Prose leaves use a sliding window of ``TARGET_TOKENS`` ≈ 550 tokens with an
``OVERLAP_TOKENS`` ≈ 70-token back-step to preserve context at chunk boundaries.
"""

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
    """A single indexable unit produced by the chunking pipeline.

    Attributes:
        chunk_id: UUID string, unique across all guidelines.
        guideline_id: Short identifier for the source guideline (e.g. "mamma").
        guideline_version: Version string extracted from the pipeline GUIDELINE_MAP.
        text: The raw text content that will be embedded and stored.
        chunk_type: One of: "section", "recommendation", "evidence", "rationale", "table".
        is_leaf: True when the chunk should be retrieved; False for parent/heading chunks.
        parent_chunk_id: UUID of the immediate parent heading chunk, or None for roots.
        root_chunk_id: UUID of the top-level (depth-1) heading ancestor, or None.
        section_path: Ordered list of section numbers from root to this chunk's heading.
        section_title: Text of the immediately enclosing heading.
        recommendation_id: Numeric id string from the "Empfehlung" label (e.g. "4.7").
        recommendation_grade: Extracted grade character: "A", "B", or "0".
        evidence_level: Extracted evidence level string (e.g. "1a", "2b").
        page_start: Physical PDF page number where the chunk begins (1-based).
        page_end: Physical PDF page number where the chunk ends (1-based).
        chunk_index_in_parent: Zero-based position among siblings under the same parent.
        source_filename: Original PDF filename, attached by ``attach_metadata``.
        is_current: False when this chunk belongs to a superseded guideline version.
    """

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
    """Generate a new UUID string for use as a chunk_id."""
    return str(uuid.uuid4())


def build_chunks(
    guideline_id: str,
    guideline_version: str,
    text: str,
    page_boundaries: list[PageBoundary] | None = None,
) -> list[Chunk]:
    """Convert cleaned guideline text into a list of indexable ``Chunk`` objects.

    Orchestrates the full structure-aware chunking pass:
    1. Calls ``detect_structure`` to classify every line into headings, recommendations,
       prose, or bibliography entries.
    2. Maintains a running section path and parent/root chain as headings are consumed.
    3. Accumulates prose lines in a buffer and flushes them as sliding-window leaf chunks
       when a structural boundary (heading, recommendation, bib entry) is encountered.
    4. Maps each chunk back to a physical PDF page via ``page_boundaries``.

    Args:
        guideline_id: Short guideline key (e.g. "mamma"), stored on every chunk.
        guideline_version: Version string stored on every chunk.
        text: Pre-cleaned full document text (output of ``parser.clean_text``).
        page_boundaries: Optional list of ``(line_start, page_number)`` tuples used to
            assign ``page_start``/``page_end`` fields.  When ``None``, page fields are left
            as ``None``.

    Returns:
        Ordered list of ``Chunk`` objects ready for embedding and upsert.
    """
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
