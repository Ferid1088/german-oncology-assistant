from unittest.mock import MagicMock
from src.tools.search_guidelines import search_guidelines_tool
from src.tools.lookup_empfehlung import lookup_empfehlung_tool


def test_search_guidelines_returns_list(mocker):
    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "abc"
    mock_chunk.text = "Empfehlung text"
    mock_chunk.score = 0.9
    mock_chunk.guideline_id = "mamma"
    mock_chunk.section_title = "Diagnose"
    mock_chunk.section_path = ["2", "2.1"]
    mock_chunk.page_start = 10
    mock_chunk.page_end = 11
    mock_chunk.page_numbers = [127, 129]
    mock_chunk.recommendation_grade = "A"
    mock_chunk.recommendation_id = "2.1"
    mock_chunk.source_filename = "mammakarzinom_v4.4.pdf"

    mocker.patch("src.tools.search_guidelines.hybrid_search", return_value=[mock_chunk])
    mocker.patch("src.tools.search_guidelines.rerank", return_value=[mock_chunk])
    mocker.patch("src.tools.search_guidelines.expand_to_parents", return_value=[mock_chunk])

    result = search_guidelines_tool(query="Screening Mammakarzinom")
    assert isinstance(result, list)
    assert result[0]["chunk_id"] == "abc"
    assert "text" in result[0]
    assert "citation" in result[0]
    assert result[0]["page_numbers"] == [127, 129]
    assert "S. 127, 129" in result[0]["citation"]


def test_lookup_empfehlung_queries_by_id(mocker):
    mock_client = mocker.MagicMock()
    mock_client.query.return_value = [{
        "chunk_id": "emp1",
        "text": "Empfehlung 2.1 Text",
        "recommendation_grade": "A",
        "evidence_level": "1a",
        "section_title": "Diagnose",
        "guideline_id": "mamma",
    }]
    result = lookup_empfehlung_tool(guideline_id="mamma", recommendation_id="2.1", client=mock_client)
    assert result["recommendation_id"] == "2.1"
    assert result["recommendation_grade"] == "A"
