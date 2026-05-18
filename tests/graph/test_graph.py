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
