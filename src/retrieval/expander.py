import os
from pymilvus import MilvusClient
from src.retrieval.search import RetrievedChunk

MILVUS_URI = os.getenv("MILVUS_URI") or "./milvus.db"
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")


def expand_to_parents(
    chunks: list[RetrievedChunk],
    client: MilvusClient | None = None,
) -> list[RetrievedChunk]:
    """For each leaf chunk that has a parent, fetch and attach parent context."""
    c = client or MilvusClient(uri=MILVUS_URI)
    parent_ids = list({chunk.parent_chunk_id for chunk in chunks if chunk.parent_chunk_id})
    parent_text_by_id: dict[str, str] = {}

    if parent_ids:
        try:
            parent_rows = c.get(
                collection_name=COLLECTION,
                ids=parent_ids,
                output_fields=["chunk_id", "text"],
            )
            parent_text_by_id = {
                row.get("chunk_id", ""): row.get("text", "")
                for row in parent_rows
                if row.get("chunk_id")
            }
        except Exception:
            parent_text_by_id = {}

    result = []
    for chunk in chunks:
        if not chunk.parent_chunk_id:
            result.append(chunk)
            continue
        try:
            parent_text = parent_text_by_id.get(chunk.parent_chunk_id, "")
            if parent_text:
                expanded = RetrievedChunk(
                    **{**vars(chunk), "text": f"{parent_text}\n\n{chunk.text}"}
                )
                result.append(expanded)
            else:
                result.append(chunk)
        except Exception:
            result.append(chunk)
    return result
