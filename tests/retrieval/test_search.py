from src.retrieval.search import hybrid_search, rrf_fuse, RetrievedChunk


def _make_chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="t",
        score=score,
        guideline_id="g",
        section_title="s",
        page_start=1,
        page_end=1,
        chunk_type="section",
        recommendation_grade="",
        section_path=[],
    )


def test_rrf_fuse_combines_results():
    dense = [_make_chunk("a", 0.9)]
    sparse = [_make_chunk("b", 0.8)]
    combined = rrf_fuse(dense, sparse, k=60)
    ids = [c.chunk_id for c in combined]
    assert "a" in ids
    assert "b" in ids


def test_rrf_fuse_boosts_chunk_in_both_lists():
    chunk_a = _make_chunk("a")
    combined_both = rrf_fuse([chunk_a], [chunk_a], k=60)
    combined_one = rrf_fuse([chunk_a], [], k=60)
    assert combined_both[0].score > combined_one[0].score
