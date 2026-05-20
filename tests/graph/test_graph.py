from unittest.mock import patch, MagicMock
import pytest


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
