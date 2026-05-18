from unittest.mock import MagicMock
from src.graph.state import RAGState
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.confidence import check_confidence
from src.graph.nodes.guardrail_input import apply_input_guardrail


def _base_state() -> RAGState:
    return RAGState(
        user_query="Welche Empfehlung gilt für das Screening?",
        session_id="test-session",
        rewritten_query="",
        metadata_filters={},
        intent="",
        retrieved_chunks=[],
        confidence=0.0,
        answer_professional="",
        answer_plain="",
        citations=[],
        disclaimer="",
        input_blocked=False,
        input_block_reason="",
        output_blocked=False,
        tool_calls_log=[],
        messages=[],
    )


def test_rewrite_query_updates_state(mocker):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Screening-Empfehlung Mammakarzinom"))]
    )
    state = _base_state()
    result = rewrite_query(state, client=mock_client)
    assert result["rewritten_query"] == "Screening-Empfehlung Mammakarzinom"


def test_confidence_high_when_chunks_present():
    state = _base_state()
    state["retrieved_chunks"] = [{"score": 0.85}, {"score": 0.80}, {"score": 0.75}]
    result = check_confidence(state)
    assert result["confidence"] > 0.5


def test_confidence_low_when_no_chunks():
    state = _base_state()
    state["retrieved_chunks"] = []
    result = check_confidence(state)
    assert result["confidence"] == 0.0


def test_input_guardrail_blocks_offtopic():
    state = _base_state()
    state["user_query"] = "Wie koche ich Spaghetti?"
    result = apply_input_guardrail(state)
    assert result["input_blocked"] is True


def test_input_guardrail_passes_medical_query():
    state = _base_state()
    result = apply_input_guardrail(state)
    assert result["input_blocked"] is False
