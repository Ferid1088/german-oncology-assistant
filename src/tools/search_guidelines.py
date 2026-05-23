"""Core RAG retrieval tool: hybrid search → rerank → parent expansion.

This is the primary tool called by the agent on every turn (forced on iteration 1).
It orchestrates the full retrieval pipeline and returns a serialisable list of
chunk dicts enriched with citation metadata ready for the answer generation node.
"""

from src.citations import format_page_reference
from src.retrieval.search import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.expander import expand_to_parents


def search_guidelines_tool(
    query: str,
    guideline_id: str | None = None,
    grade: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Core RAG retrieval tool. Returns ranked chunks with citation metadata.
    Suitable for use as a LangGraph tool function.
    """
    # Fetch more candidates than needed so the reranker has a larger pool
    # to work with; capped at 20 to keep Milvus latency predictable.
    candidate_pool_size = min(20, max(top_k, top_k * 3))

    candidates = hybrid_search(
        query=query,
        guideline_id=guideline_id,
        grade_filter=grade,
        top_k=candidate_pool_size,
    )
    reranked = rerank(query=query, chunks=candidates, top_k=top_k)
    expanded = expand_to_parents(reranked)

    return [
        {
            "chunk_id": c.chunk_id,
            "citation_label": f"[{i + 1}]",
            "text": c.text,
            "score": round(c.score, 4),
            "guideline_id": c.guideline_id,
            "section_title": c.section_title,
            "section_path": c.section_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "page_numbers": c.page_numbers or [],
            "chunk_type": c.chunk_type,
            "recommendation_grade": c.recommendation_grade,
            "recommendation_id": c.recommendation_id,
            "evidence_level": c.evidence_level,
            "source_filename": c.source_filename,
            "contextual_header": c.contextual_header,
            "parent_chunk_id": c.parent_chunk_id,
            "reference_ids": c.reference_ids or [],
            "citation": f"{c.guideline_id.upper()} § {'.'.join(c.section_path)} ({format_page_reference(c.page_numbers, c.page_start, c.page_end, short=True)})"
            if format_page_reference(c.page_numbers, c.page_start, c.page_end, short=True)
            else c.guideline_id.upper(),
        }
        for i, c in enumerate(expanded)
    ]
