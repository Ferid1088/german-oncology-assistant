from src.indexer.chunker import Chunk
from collections import defaultdict


def attach_metadata(
    chunks: list[Chunk],
    source_filename: str,
    is_current: bool = True,
) -> list[Chunk]:
    """Attach deterministic structural metadata fields that require cross-chunk context."""
    parent_counters: dict[str | None, int] = defaultdict(int)
    for chunk in chunks:
        chunk.source_filename = source_filename
        chunk.is_current = is_current
        if chunk.is_leaf and chunk.chunk_index_in_parent is None:
            # Only assign index if not already set by the chunker (e.g. empfehlung leaves)
            chunk.chunk_index_in_parent = parent_counters[chunk.parent_chunk_id]
            parent_counters[chunk.parent_chunk_id] += 1
    return chunks
