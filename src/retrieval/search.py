import os
import json
from dataclasses import dataclass
from pymilvus import MilvusClient
from src.indexer.embedder import embed_texts

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")
TOP_K_DENSE = 20
TOP_K_SPARSE = 20


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    guideline_id: str
    section_title: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    chunk_type: str
    recommendation_grade: str
    recommendation_id: str = ""
    parent_chunk_id: str = ""
    source_filename: str = ""
    contextual_header: str = ""


def rrf_fuse(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion over two ranked lists."""
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        by_id[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        by_id[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        RetrievedChunk(**{**vars(by_id[cid]), "score": sc})
        for cid, sc in ranked
    ]


def _milvus_results_to_chunks(results: list[dict]) -> list[RetrievedChunk]:
    chunks = []
    for r in results:
        e = r.get("entity", r)
        chunks.append(RetrievedChunk(
            chunk_id=e.get("chunk_id", ""),
            text=e.get("text", ""),
            score=r.get("distance", 0.0),
            guideline_id=e.get("guideline_id", ""),
            section_title=e.get("section_title", ""),
            section_path=json.loads(e.get("section_path", "[]")),
            page_start=e.get("page_start"),
            page_end=e.get("page_end"),
            chunk_type=e.get("chunk_type", ""),
            recommendation_grade=e.get("recommendation_grade", ""),
            recommendation_id=e.get("recommendation_id", ""),
            parent_chunk_id=e.get("parent_chunk_id", ""),
            source_filename=e.get("source_filename", ""),
            contextual_header=e.get("contextual_header", ""),
        ))
    return chunks


def hybrid_search(
    query: str,
    guideline_id: str | None = None,
    grade_filter: str | None = None,
    chunk_type_filter: str | None = None,
    top_k: int = 20,
    client: MilvusClient | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 search with RRF fusion and optional metadata filter."""
    c = client or MilvusClient(uri=MILVUS_URI)
    vector = embed_texts([query])[0]

    output_fields = [
        "chunk_id", "text", "guideline_id", "section_title", "section_path",
        "page_start", "page_end", "chunk_type", "recommendation_grade",
        "recommendation_id", "parent_chunk_id", "source_filename", "contextual_header",
    ]

    # Build metadata filter expression
    filters = ["is_leaf == true"]
    if guideline_id:
        filters.append(f'guideline_id == "{guideline_id}"')
    if grade_filter:
        filters.append(f'recommendation_grade == "{grade_filter}"')
    if chunk_type_filter:
        filters.append(f'chunk_type == "{chunk_type_filter}"')
    expr = " and ".join(filters) if filters else ""

    # Dense search
    dense_raw = c.search(
        collection_name=COLLECTION,
        data=[vector],
        anns_field="dense_vector",
        limit=top_k,
        filter=expr,
        output_fields=output_fields,
    )
    dense_chunks = _milvus_results_to_chunks(dense_raw[0] if dense_raw else [])

    # BM25 sparse search — Milvus 2.4+ supports full-text search
    # Fallback: use dense only if BM25 not configured
    try:
        sparse_raw = c.search(
            collection_name=COLLECTION,
            data=[query],
            anns_field="sparse_vector",
            search_params={"metric_type": "BM25"},
            limit=top_k,
            filter=expr,
            output_fields=output_fields,
        )
        sparse_chunks = _milvus_results_to_chunks(sparse_raw[0] if sparse_raw else [])
    except Exception:
        sparse_chunks = []  # BM25 not configured — use dense only

    return rrf_fuse(dense_chunks, sparse_chunks)[:top_k]
