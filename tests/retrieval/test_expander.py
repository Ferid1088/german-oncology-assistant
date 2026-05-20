from unittest.mock import MagicMock

from src.retrieval.expander import expand_to_parents
from src.retrieval.search import RetrievedChunk


def _make_chunk(chunk_id: str, parent_chunk_id: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"leaf {chunk_id}",
        score=0.9,
        guideline_id="g",
        section_title="s",
        section_path=[],
        page_start=1,
        page_end=1,
        chunk_type="section",
        recommendation_grade="",
        parent_chunk_id=parent_chunk_id,
    )


def test_expand_to_parents_batches_parent_fetches():
    client = MagicMock()
    client.get.return_value = [
        {"chunk_id": "p1", "text": "Parent 1"},
        {"chunk_id": "p2", "text": "Parent 2"},
    ]

    chunks = [
        _make_chunk("c1", parent_chunk_id="p1"),
        _make_chunk("c2", parent_chunk_id="p2"),
        _make_chunk("c3", parent_chunk_id="p1"),
    ]

    expanded = expand_to_parents(chunks, client=client)

    client.get.assert_called_once()
    assert expanded[0].text == "Parent 1\n\nleaf c1"
    assert expanded[1].text == "Parent 2\n\nleaf c2"
    assert expanded[2].text == "Parent 1\n\nleaf c3"