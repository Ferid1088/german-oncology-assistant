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
    candidates = hybrid_search(
        query=query,
        guideline_id=guideline_id,
        grade_filter=grade,
        top_k=20,
    )
    reranked = rerank(query=query, chunks=candidates, top_k=top_k)
    expanded = expand_to_parents(reranked)

    return [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "score": round(c.score, 4),
            "guideline_id": c.guideline_id,
            "section_title": c.section_title,
            "section_path": c.section_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "recommendation_grade": c.recommendation_grade,
            "recommendation_id": c.recommendation_id,
            "source_filename": c.source_filename,
            "citation": f"{c.guideline_id.upper()} § {'.'.join(c.section_path)} (S. {c.page_start}–{c.page_end})"
            if c.page_start
            else c.guideline_id.upper(),
        }
        for c in expanded
    ]
