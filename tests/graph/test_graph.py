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


def test_clarification_response_returns_single_plain_message():
    from src.graph.graph import _clarification_response

    result = _clarification_response(
        {
            "clarification_rationale": "Die klinische Situation ist noch zu unspezifisch.",
            "expected_clarification": "Bitte präzisieren Sie, ob es um eine adjuvante oder metastasierte Situation geht.",
        }
    )

    assert result["answer_professional"].startswith("Ich brauche vor der Leitlinienrecherche")
    assert "Die klinische Situation ist noch zu unspezifisch." in result["answer_professional"]
    assert "Bitte präzisieren Sie, ob es um eine adjuvante oder metastasierte Situation geht." in result["answer_professional"]
    assert result["answer_plain"] == ""


def test_route_after_output_uses_external_search_when_not_blocked():
    from src.graph.graph import _route_after_output

    assert _route_after_output({"input_blocked": False, "output_blocked": False, "requires_clarification": False}) == "external_search"
    assert _route_after_output({"input_blocked": False, "output_blocked": True, "requires_clarification": False}) == "end"
