import os
from pymilvus import MilvusClient
from src.retrieval.search import RetrievedChunk

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")


def expand_to_parents(
    chunks: list[RetrievedChunk],
    client: MilvusClient | None = None,
) -> list[RetrievedChunk]:
    """For each leaf chunk that has a parent, fetch and attach parent context."""
    c = client or MilvusClient(uri=MILVUS_URI)
    result = []
    for chunk in chunks:
        if not chunk.parent_chunk_id:
            result.append(chunk)
            continue
        try:
            parent_rows = c.get(
                collection_name=COLLECTION,
                ids=[chunk.parent_chunk_id],
                output_fields=["text"],
            )
            if parent_rows:
                parent_text = parent_rows[0].get("text", "")
                expanded = RetrievedChunk(
                    **{**vars(chunk), "text": f"{parent_text}\n\n{chunk.text}"}
                )
                result.append(expanded)
            else:
                result.append(chunk)
        except Exception:
            result.append(chunk)
    return result
