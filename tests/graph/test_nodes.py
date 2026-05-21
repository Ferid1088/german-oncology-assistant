from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.state import RAGState
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.turn_router import route_turn
from src.graph.nodes.agent import run_agent
from src.graph.nodes.answer import generate_answer
from src.graph.nodes.confidence import check_confidence
from src.graph.nodes.guardrail_input import apply_input_guardrail
from src.graph.nodes.guardrail_output import apply_output_guardrail
import src.api.routes.chat as chat_module
from src.api.conversation_store import ConversationStore


def _base_state() -> RAGState:
    return RAGState(
        user_query="Welche Empfehlung gilt für das Screening?",
        session_id="test-session",
        rewritten_query="",
        metadata_filters={},
        intent="",
        query_decomposition=[],
        requires_clarification=False,
        missing_clinical_dimensions=[],
        clarification_rationale=None,
        expected_clarification=None,
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
        safety_warning=None,
        safety_explanation=None,
        safety_title=None,
        tool_calls_log=[],
        rag_trace=[],
        token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "currency": "USD", "calls": []},
        external_search_snippets=[],
        turn_intents=[],
        followup_routing="retrieve",
        prior_answer_professional="",
        prior_answer_plain="",
        prior_citations=[],
        prior_retrieved_chunks=[],
        prior_rewritten_query="",
        prior_rag_trace=[],
        prior_external_search_snippets=[],
        messages=[],
    )


def test_rewrite_query_updates_state(mocker):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"rewritten_query":"Screening-Empfehlung Mammakarzinom","guideline_id":"","grade":"","chunk_type":"","intent":"factual","requires_clarification":false,"missing_clinical_dimensions":[],"clarification_rationale":null,"expected_clarification":null}'))]
    )
    state = _base_state()
    result = rewrite_query(state, client=mock_client)
    assert result["rewritten_query"] == "Screening-Empfehlung Mammakarzinom"
    assert result["requires_clarification"] is False


def test_rewrite_query_detects_clarification_need():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"rewritten_query":"Therapie Empfehlung Mammakarzinom Leitlinie","guideline_id":"mamma","grade":"","chunk_type":"recommendation","intent":"recommendation","requires_clarification":true,"missing_clinical_dimensions":["disease_stage","therapy_setting","molecular_subtype"],"clarification_rationale":"Die Anfrage ist klinisch zu unspezifisch.","expected_clarification":"Bitte präzisieren Sie, ob es um eine adjuvante, neoadjuvante oder metastasierte Situation geht."}'))]
    )

    result = rewrite_query(_base_state(), client=mock_client)

    assert result["requires_clarification"] is True
    assert result["missing_clinical_dimensions"] == ["disease_stage", "therapy_setting", "molecular_subtype"]
    assert result["expected_clarification"].startswith("Bitte präzisieren Sie")


def test_rewrite_query_allows_only_one_clarification_turn():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"rewritten_query":"Chemotherapie metastasiertes HER2-positives Mammakarzinom Leitlinie","guideline_id":"mamma","grade":"","chunk_type":"recommendation","intent":"recommendation","requires_clarification":true,"missing_clinical_dimensions":["line_of_therapy","patient_subgroup"],"clarification_rationale":"Es fehlen weitere Details.","expected_clarification":"Bitte nennen Sie noch die Therapielinie."}'))]
    )

    state = _base_state()
    state["user_query"] = "HER2-positiv"
    state["messages"] = [
        HumanMessage(content="Was sind die Empfehlungen zur Chemotherapie beim Mammakarzinom?"),
        AIMessage(content="Fachliche Antwort:\nIch brauche vor der Leitlinienrecherche noch eine Präzisierung Ihrer Frage.\n\nBitte präzisieren Sie, ob es um eine adjuvante, neoadjuvante oder metastasierte Situation geht."),
        HumanMessage(content="metastasiert"),
    ]

    result = rewrite_query(state, client=mock_client)

    assert result["requires_clarification"] is False
    assert result["missing_clinical_dimensions"] == []
    assert result["clarification_rationale"] is None
    assert result["expected_clarification"] is None


def test_rewrite_query_blocks_repeat_clarification_from_prior_answer_memory():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"rewritten_query":"neoadjuvante Chemotherapie Mammakarzinom Leitlinie","guideline_id":"mamma","grade":"","chunk_type":"recommendation","intent":"recommendation","requires_clarification":true,"missing_clinical_dimensions":["molecular_subtype","biomarker_status"],"clarification_rationale":"Es fehlen weitere Details.","expected_clarification":"Bitte nennen Sie noch den Subtyp."}'))]
    )

    state = _base_state()
    state["user_query"] = "neoadjuvant"
    state["messages"] = []
    state["prior_answer_professional"] = (
        "Ich brauche vor der Leitlinienrecherche noch eine Präzisierung Ihrer Frage.\n\n"
        "Bitte präzisieren Sie, für welche klinische Situation Sie Empfehlungen suchen."
    )

    result = rewrite_query(state, client=mock_client)

    assert result["requires_clarification"] is False
    assert result["missing_clinical_dimensions"] == []
    assert result["clarification_rationale"] is None
    assert result["expected_clarification"] is None


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
            "page_numbers": [1],
        }
    ]

    result = generate_answer(state, client=mock_client)

    assert result["answer_professional"] == "Kurzfassung in 2 Sätzen [1]. Zweiter Satz [1]."
    assert result["answer_plain"] == ""
    assert result["citations"][0]["label"] == "[1]"


def test_in_memory_session_fallback_persists_prior_context(tmp_path, monkeypatch):
    session_id = "memory-test"
    store = ConversationStore(tmp_path / "chat-memory.db")
    monkeypatch.setattr(chat_module, "get_conversation_store", lambda: store)

    first = chat_module._load_session_memory(session_id)
    assert first["messages"] == []

    final_state = {
        "answer_professional": "Vorherige fachliche Antwort",
        "answer_plain": "Vorherige einfache Antwort",
        "citations": [{"label": "[1]"}],
        "retrieved_chunks": [{"chunk_id": "c1", "citation": "[1]", "text": "abc"}],
        "rag_trace": [{"name": "rewrite", "status": "ok", "details": {"rewritten_query": "empfehlungsgrade leitlinie"}}],
    }
    chat_module._save_session_memory(session_id, final_state, "Welche Empfehlungsgrade gibt es?")

    second = chat_module._load_session_memory(session_id)
    assert len(second["messages"]) == 2
    assert "Vorherige fachliche Antwort" in str(second["messages"][1].content)
    assert "Vorherige einfache Antwort" in str(second["messages"][1].content)
    assert second["prior_answer_professional"] == "Vorherige fachliche Antwort"
    assert second["prior_retrieved_chunks"] == final_state["retrieved_chunks"]
    assert second["prior_rewritten_query"] == "empfehlungsgrade leitlinie"


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


def test_output_guardrail_adds_warning_for_patient_specific_queries():
    state = _base_state()
    state["user_query"] = "Welche adjuvante Therapie gilt für meine Patientin mit HER2-positivem Mammakarzinom?"
    state["retrieved_chunks"] = [{"text": "Allgemeine Empfehlung", "citation": "MAMMA 1", "section_title": "Therapie"}]
    state["answer_professional"] = "Allgemeine Leitlinienempfehlung [1]."

    result = apply_output_guardrail(state)

    assert result["output_blocked"] is False
    assert result["safety_warning"] is not None
    assert "general guideline" in result["safety_warning"].lower()


def test_output_guardrail_blocks_unsupported_dosage_requests():
    state = _base_state()
    state["user_query"] = "Welche Dosis in mg soll ich bei dieser Patientin geben?"
    state["retrieved_chunks"] = [{"text": "Therapieempfehlung ohne Dosisangabe", "citation": "MAMMA 2", "section_title": "Therapie"}]
    state["answer_professional"] = "Antwort mit möglicher Dosis."

    result = apply_output_guardrail(state)

    assert result["output_blocked"] is True
    assert result["safety_warning"] is not None
    assert result["citations"] == []


def test_output_guardrail_allows_dosage_when_directly_grounded():
    state = _base_state()
    state["user_query"] = "Welche Dosis ist in der Leitlinie genannt?"
    state["retrieved_chunks"] = [
        {
            "text": "Empfohlen werden 80 mg/m² alle 3 Wochen im dargestellten Schema.",
            "citation": "MAMMA 3",
            "section_title": "Dosierung",
        }
    ]
    state["answer_professional"] = "Die Leitlinie nennt 80 mg/m² [1]."

    result = apply_output_guardrail(state)

    assert result["output_blocked"] is False
    assert result["safety_warning"] is None
