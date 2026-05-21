from fastapi.testclient import TestClient

import src.api.routes.analytics as analytics_module
from src.api.auth import reset_auth_config_cache
from src.api.conversation_store import ConversationStore
from src.api.main import app
from src.api.rate_limit import _RATE_LIMITER


def test_analytics_overview_endpoint_returns_aggregated_metrics(tmp_path, monkeypatch):
    store = ConversationStore(tmp_path / "analytics.db")
    monkeypatch.setenv("API_KEYS", "alpha")
    reset_auth_config_cache()
    _RATE_LIMITER._events.clear()
    monkeypatch.setattr(analytics_module, "get_conversation_store", lambda: store)

    store.append_turn(
        conversation_id="conv-1",
        user_query="Was ist empfohlen?",
        final_state={
            "answer_professional": "Fachliche Antwort",
            "answer_plain": "Einfache Antwort",
            "citations": [{"guideline_id": "mamma", "title": "Quelle 1"}],
            "retrieved_chunks": [],
            "tool_calls_log": [{"tool": "search_guidelines"}],
            "rag_trace": [{"name": "rewrite", "status": "ok"}],
            "token_usage": {"total_tokens": 12, "cost_usd": 0.0001, "calls": []},
            "external_search_snippets": [],
        },
        combined_answer="Fachliche Antwort\n\nEinfache Antwort",
    )
    store.append_turn(
        conversation_id="conv-2",
        user_query="Brauche ich eine Websuche?",
        final_state={
            "answer_professional": "Antwort 2",
            "answer_plain": "Plain 2",
            "citations": [{"guideline_id": "lunge", "title": "Quelle 2"}],
            "retrieved_chunks": [],
            "tool_calls_log": [{"tool": "web_search"}],
            "rag_trace": [{"name": "retrieve", "status": "ok"}, {"name": "external_search", "status": "ok"}],
            "token_usage": {"total_tokens": 18, "cost_usd": 0.0002, "calls": []},
            "external_search_snippets": [{"title": "Result"}],
        },
        combined_answer="Antwort 2\n\nPlain 2",
    )

    client = TestClient(app)
    response = client.get("/analytics/overview?session_id=conv-2", headers={"X-API-Key": "alpha"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["overview"]["total_conversations"] == 2
    assert payload["overview"]["total_questions"] == 2
    assert payload["overview"]["total_answers"] == 2
    assert payload["overview"]["total_tokens"] == 30
    assert payload["overview"]["total_cost_usd"] == 0.0003
    assert payload["current_session"]["session_id"] == "conv-2"
    assert payload["distributions"]["tools"][0]["count"] == 1
    assert {item["label"] for item in payload["distributions"]["guidelines"]} == {"mamma", "lunge"}