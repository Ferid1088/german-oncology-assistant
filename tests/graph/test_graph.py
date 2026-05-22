import pytest
from unittest.mock import MagicMock


def _make_mock_client(content="Antwort"):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content, tool_calls=None))]
    )
    return client


def test_graph_compiles():
    from src.graph.graph import build_graph
    graph = build_graph()
    assert graph is not None


def test_route_after_rewrite_branches_to_clarification_when_needed():
    from src.graph.graph import _route_after_rewrite

    assert _route_after_rewrite({"requires_clarification": True}) == "clarification"
    assert _route_after_rewrite({"requires_clarification": False}) == "turn_router"


def test_route_after_rewrite_skips_clarification_after_followup_turn():
    from src.graph.graph import _route_after_rewrite

    assert _route_after_rewrite({"requires_clarification": False}) == "turn_router"


def test_route_after_rewrite_branches_to_repeat_answer_when_rewritten_query_matches_prior():
    from src.graph.graph import _route_after_rewrite

    result = _route_after_rewrite(
        {
            "requires_clarification": False,
            "rewritten_query": "Therapie Mammakarzinom Leitlinie",
            "prior_rewritten_query": "therapie mammakarzinom leitlinie",
            "prior_answer_professional": "Vorherige Antwort",
            "prior_answer_plain": "",
        }
    )

    assert result == "repeat_answer"


def test_repeat_previous_answer_response_reuses_prior_answer_without_new_attempts():
    from src.graph.graph import _repeat_previous_answer_response
    from src.graph.nodes.answer import DISCLAIMER

    result = _repeat_previous_answer_response(
        {
            "rewritten_query": "Therapie Mammakarzinom Leitlinie",
            "prior_rewritten_query": "therapie mammakarzinom leitlinie",
            "prior_answer_professional": "Vorherige fachliche Antwort",
            "prior_answer_plain": "Vorherige einfache Antwort",
            "prior_citations": [{"label": "[1]"}],
            "prior_retrieved_chunks": [{"chunk_id": "c1"}],
            "prior_external_search_snippets": [{"title": "Snippet"}],
            "prior_rag_trace": [{"name": "answer", "status": "ok"}],
            "rag_trace": [],
        }
    )

    assert result["answer_professional"] == "Vorherige fachliche Antwort"
    assert result["answer_plain"] == "Vorherige einfache Antwort"
    assert result["citations"][0]["label"] == "[1]"
    assert result["external_search_snippets"][0]["title"] == "Snippet"
    assert result["tool_calls_log"] == []
    assert result["disclaimer"] == DISCLAIMER
    assert result["rag_trace"][-1]["name"] == "repeat_answer"


def test_clarification_response_returns_single_plain_message():
    from src.graph.graph import _clarification_response

    result = _clarification_response(
        {
            "missing_clinical_dimensions": ["disease_stage", "therapy_setting"],
            "clarification_rationale": "Die klinische Situation ist noch zu unspezifisch.",
            "expected_clarification": "Bitte präzisieren Sie, ob es um eine adjuvante oder metastasierte Situation geht.",
        }
    )

    assert result["requires_clarification"] is True
    assert result["missing_clinical_dimensions"] == ["disease_stage", "therapy_setting"]
    assert result["expected_clarification"].startswith("Bitte präzisieren Sie")
    assert result["answer_professional"].startswith("Ich brauche vor der Leitlinienrecherche")
    assert "Die klinische Situation ist noch zu unspezifisch." in result["answer_professional"]
    assert "Bitte präzisieren Sie, ob es um eine adjuvante oder metastasierte Situation geht." in result["answer_professional"]
    assert result["answer_plain"] == ""


def test_route_after_output_uses_external_search_when_not_blocked():
    from src.graph.graph import _route_after_output

    assert _route_after_output({"input_blocked": False, "output_blocked": False, "requires_clarification": False}) == "external_search"
    assert _route_after_output({"input_blocked": False, "output_blocked": True, "requires_clarification": False}) == "end"
