from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.state import RAGState
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.turn_router import route_turn
from src.graph.nodes.agent import run_agent
from src.graph.nodes.answer import generate_answer
from src.graph.nodes.confidence import check_confidence
from src.graph.nodes.guardrail_input import apply_input_guardrail
from src.api.routes.chat import _load_session_memory, _save_session_memory, _session_memory


def _base_state() -> RAGState:
    return RAGState(
        user_query="Welche Empfehlung gilt für das Screening?",
        session_id="test-session",
        rewritten_query="",
        metadata_filters={},
        intent="",
        query_decomposition=[],
        user_role="user",
        allowed_sources=["guidelines"],
        retrieved_chunks=[],
        confidence=0.0,
        escalation_reason="",
        answer_professional="",
        answer_plain="",
        citations=[],
        disclaimer="",
        input_blocked=False,
        input_block_reason="",
        output_blocked=False,
        redacted_query="Welche Empfehlung gilt für das Screening?",
        tool_calls_log=[],
        turn_intents=[],
        followup_routing="retrieve",
        prior_answer_professional="",
        prior_answer_plain="",
        prior_citations=[],
        prior_retrieved_chunks=[],
        messages=[],
    )


def test_rewrite_query_updates_state(mocker):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"rewritten_query":"Screening-Empfehlung Mammakarzinom","guideline_id":"","grade":"","chunk_type":"","intent":"factual"}'))]
    )
    state = _base_state()
    result = rewrite_query(state, client=mock_client)
    assert result["rewritten_query"] == "Screening-Empfehlung Mammakarzinom"


def test_turn_router_supports_combined_intents():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"turn_intents":["clarify","simplify"],"followup_routing":"memory"}'))]
    )
    state = _base_state()
    state["messages"] = [
        {"role": "user", "content": "Welche Empfehlungsgrade gibt es?"},
        {"role": "assistant", "content": "Es gibt A, B und 0."},
    ]
    state["user_query"] = "Kannst du das einfacher erklären?"
    result = route_turn(state, client=mock_client)
    assert result["turn_intents"] == ["clarify", "simplify"]
    assert result["followup_routing"] == "memory"


def test_turn_router_supports_langchain_message_objects():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"turn_intents":["clarify","expand"],"followup_routing":"memory"}'))]
    )
    state = _base_state()
    state["messages"] = [
        HumanMessage(content="Welche Empfehlungsgrade gibt es?"),
        AIMessage(content="Es gibt A, B und 0."),
    ]
    state["user_query"] = "Kannst du das genauer erklären?"
    result = route_turn(state, client=mock_client)
    assert result["turn_intents"] == ["clarify", "expand"]
    assert result["followup_routing"] == "memory"


def test_turn_router_heuristic_routes_summary_followup_to_memory():
    state = _base_state()
    state["messages"] = [
        HumanMessage(content="Wie ist die Empfehlung?"),
        AIMessage(content="Fachliche Antwort:\nEine längere Antwort [1]."),
    ]
    state["user_query"] = "gebe deine antwort nur in 2 sätze"

    result = route_turn(state, client=MagicMock())

    assert result["followup_routing"] == "memory"
    assert "refine" in result["turn_intents"]


def test_run_agent_reuses_prior_chunks_for_memory_route():
    state = _base_state()
    state["followup_routing"] = "memory"
    state["prior_retrieved_chunks"] = [{"chunk_id": "c1", "citation": "[1]", "text": "abc"}]
    result = run_agent(state, client=MagicMock())
    assert result["retrieved_chunks"] == state["prior_retrieved_chunks"]


def test_generate_answer_memory_followup_rewrites_prior_answer(mocker):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"answer_professional":"Kurzfassung in 2 Sätzen [1]. Zweiter Satz [1].","answer_plain":""}'))]
    )

    state = _base_state()
    state["followup_routing"] = "memory"
    state["turn_intents"] = ["simplify", "refine"]
    state["user_query"] = "gebe deine antwort nur in 2 sätze"
    state["prior_answer_professional"] = "Lange fachliche Antwort [1]. Noch ein Satz [1]. Dritter Satz [1]."
    state["prior_answer_plain"] = ""
    state["prior_citations"] = [
        {
            "label": "[1]",
            "chunk_id": "c1",
            "citation": "MAMMA § 1",
            "source_filename": "x.pdf",
            "section_path": [],
            "page_start": 1,
            "page_end": 1,
            "section_title": "s",
            "guideline_id": "mamma",
            "recommendation_id": "",
            "recommendation_grade": "A",
            "evidence_level": "",
            "reference_ids": [],
            "contextual_header": "",
            "parent_chunk_id": "",
            "is_opinion": False,
        }
    ]

    result = generate_answer(state, client=mock_client)

    assert result["answer_professional"] == "Kurzfassung in 2 Sätzen [1]. Zweiter Satz [1]."
    assert result["answer_plain"] == ""
    assert result["citations"][0]["label"] == "[1]"


def test_in_memory_session_fallback_persists_prior_context():
    session_id = "memory-test"
    _session_memory.pop(session_id, None)

    first = _load_session_memory(session_id)
    assert first["messages"] == []

    final_state = {
        "answer_professional": "Vorherige fachliche Antwort",
        "answer_plain": "Vorherige einfache Antwort",
        "citations": [{"label": "[1]"}],
        "retrieved_chunks": [{"chunk_id": "c1", "citation": "[1]", "text": "abc"}],
    }
    _save_session_memory(session_id, final_state, "Welche Empfehlungsgrade gibt es?")

    second = _load_session_memory(session_id)
    assert len(second["messages"]) == 2
    assert "Vorherige fachliche Antwort" in str(second["messages"][1].content)
    assert "Vorherige einfache Antwort" in str(second["messages"][1].content)
    assert second["prior_answer_professional"] == "Vorherige fachliche Antwort"
    assert second["prior_retrieved_chunks"] == final_state["retrieved_chunks"]


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
    assert result["input_blocked"] is False


def test_input_guardrail_passes_medical_query():
    state = _base_state()
    result = apply_input_guardrail(state)
    assert result["input_blocked"] is False
