import src.retrieval.reranker as reranker_mod
from src.retrieval.search import RetrievedChunk


def _make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.1,
        guideline_id="g",
        section_title="s",
        section_path=[],
        page_start=1,
        page_end=1,
        chunk_type="section",
        recommendation_grade="",
    )


def test_rerank_limits_candidate_pool_and_returns_top_k(monkeypatch):
    class DummyReranker:
        def __init__(self):
            self.calls = []

        def predict(self, pairs, show_progress_bar=False, batch_size=None):
            self.calls.append({"pairs": pairs, "batch_size": batch_size})
            return [0.10, 0.90, 0.40]

    dummy = DummyReranker()
    monkeypatch.setattr(reranker_mod, "_RERANKER_AVAILABLE", True)
    monkeypatch.setattr(reranker_mod, "RERANKER_CANDIDATES", 3)
    monkeypatch.setattr(reranker_mod, "RERANKER_BATCH_SIZE", 4)
    monkeypatch.setattr(reranker_mod, "_get_reranker", lambda: dummy)

    chunks = [
        _make_chunk("a", "one"),
        _make_chunk("b", "two"),
        _make_chunk("c", "three"),
        _make_chunk("d", "four"),
    ]

    ranked = reranker_mod.rerank(query="q", chunks=chunks, top_k=2)

    assert [chunk.chunk_id for chunk in ranked] == ["b", "c"]
    assert len(dummy.calls) == 1
    assert len(dummy.calls[0]["pairs"]) == 3
    assert dummy.calls[0]["batch_size"] == 4