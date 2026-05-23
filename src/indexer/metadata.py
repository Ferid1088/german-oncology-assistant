"""Post-processing step that attaches cross-chunk metadata to all ``Chunk`` objects.

Runs after ``build_chunks`` and before embedding.  Assigns ``source_filename`` and
``is_current`` to every chunk, and fills in ``chunk_index_in_parent`` for leaf chunks
whose index was not already set by the chunker (e.g. prose leaves, as opposed to
recommendation leaves which receive ``None`` from the chunker intentionally).
"""

from src.indexer.chunker import Chunk
from collections import defaultdict


def attach_metadata(
    chunks: list[Chunk],
    source_filename: str,
    is_current: bool = True,
) -> list[Chunk]:
    """Attach deterministic structural metadata fields that require cross-chunk context.

    Must be called after ``build_chunks`` and before ``embedder.embed_chunks``.

    Args:
        chunks: List of ``Chunk`` objects produced by ``build_chunks``.
        source_filename: Original PDF filename to record on every chunk.
        is_current: Set to ``False`` when indexing a superseded guideline version so
            that retrieval can filter it out without dropping the collection.

    Returns:
        The same list of chunks, mutated in-place with metadata applied.
    """
    parent_counters: dict[str | None, int] = defaultdict(int)
    for chunk in chunks:
        chunk.source_filename = source_filename
        chunk.is_current = is_current
        if chunk.is_leaf and chunk.chunk_index_in_parent is None:
            # Assign sibling index only when the chunker left it unset (prose leaves).
            # Recommendation leaves intentionally receive None from the chunker.
            chunk.chunk_index_in_parent = parent_counters[chunk.parent_chunk_id]
            parent_counters[chunk.parent_chunk_id] += 1
    return chunks
