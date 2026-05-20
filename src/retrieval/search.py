import os
import json
import logging
import time
from dataclasses import dataclass
from pymilvus import MilvusClient
from src.indexer.embedder import embed_texts
from src.retrieval import bm25 as _bm25_mod

log = logging.getLogger(__name__)

MILVUS_URI = os.getenv("MILVUS_URI") or "./milvus.db"
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")
TOP_K_DENSE = 20


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
    evidence_level: str = ""
    parent_chunk_id: str = ""
    source_filename: str = ""
    contextual_header: str = ""
    reference_ids: list[str] | None = None


def _safe_json_loads(value, default):
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


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
            section_path=_safe_json_loads(e.get("section_path", "[]"), []),
            page_start=e.get("page_start"),
            page_end=e.get("page_end"),
            chunk_type=e.get("chunk_type", ""),
            recommendation_grade=e.get("recommendation_grade", ""),
            recommendation_id=e.get("recommendation_id", ""),
            evidence_level=e.get("evidence_level", ""),
            parent_chunk_id=e.get("parent_chunk_id", ""),
            source_filename=e.get("source_filename", ""),
            contextual_header=e.get("contextual_header", ""),
            reference_ids=_safe_json_loads(e.get("reference_ids", "[]"), []),
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
    """Dense vector search with optional metadata filter."""
    c = client or MilvusClient(uri=MILVUS_URI)
    vector = embed_texts([query])[0]

    output_fields = [
        "chunk_id", "text", "guideline_id", "section_title", "section_path",
        "page_start", "page_end", "chunk_type", "recommendation_grade",
        "recommendation_id", "evidence_level", "parent_chunk_id", "source_filename", "contextual_header",
        "reference_ids",
    ]

    filters = ["is_leaf == true"]
    if guideline_id:
        filters.append(f'guideline_id == "{guideline_id}"')
    if grade_filter:
        filters.append(f'recommendation_grade == "{grade_filter}"')
    if chunk_type_filter:
        filters.append(f'chunk_type == "{chunk_type_filter}"')
    expr = " and ".join(filters) if filters else ""

    if not c.has_collection(COLLECTION):
        log.warning("Collection '%s' not found — run the indexer first.", COLLECTION)
        return []

    for attempt in range(5):
        try:
            c.load_collection(COLLECTION)
            break
        except Exception as e:
            if attempt == 4:
                raise
            log.warning("load_collection attempt %d failed (%s), retrying...", attempt + 1, e)
            time.sleep(3)

    dense_raw = c.search(
        collection_name=COLLECTION,
        data=[vector],
        anns_field="dense_vector",
        limit=top_k,
        filter=expr,
        output_fields=output_fields,
    )
    general_chunks = _milvus_results_to_chunks(dense_raw[0] if dense_raw else [])

    # Second dense search restricted to recommendation chunks so that terse
    # empfehlung blocks aren't buried by longer prose on pure cosine similarity.
    rec_chunks: list[RetrievedChunk] = []
    if not chunk_type_filter:
        rec_filters = [f for f in filters if 'chunk_type' not in f]
        rec_filters.append('chunk_type == "recommendation"')
        rec_expr = " and ".join(rec_filters)
        rec_raw = c.search(
            collection_name=COLLECTION,
            data=[vector],
            anns_field="dense_vector",
            limit=top_k,
            filter=rec_expr,
            output_fields=output_fields,
        )
        rec_chunks = _milvus_results_to_chunks(rec_raw[0] if rec_raw else [])

    # BM25 sparse retrieval — scores exact keyword matches (drug names, rec IDs, etc.)
    bm25_chunks = _bm25_mod.bm25_search(query, top_k=top_k, guideline_id=guideline_id)

    # Three-way RRF: general dense + recommendation dense + BM25
    fused = rrf_fuse(rrf_fuse(general_chunks, rec_chunks), bm25_chunks)[:top_k]

    # BM25 stubs carry no metadata. Build a lookup from the dense results so we
    # can transplant the full metadata onto any chunk that BM25 promoted in rank.
    # Chunks that appeared only in BM25 (not in any dense result) are dropped —
    # they have no metadata we can attach and would appear as empty citations.
    dense_by_id: dict[str, RetrievedChunk] = {}
    for ch in general_chunks:
        dense_by_id[ch.chunk_id] = ch
    for ch in rec_chunks:
        dense_by_id.setdefault(ch.chunk_id, ch)

    fused = [
        RetrievedChunk(**{**vars(dense_by_id[ch.chunk_id]), "score": ch.score})
        if ch.chunk_id in dense_by_id else ch
        for ch in fused
        if ch.chunk_id in dense_by_id or ch.text  # drop BM25-only stubs with no metadata
    ]

    return fused
