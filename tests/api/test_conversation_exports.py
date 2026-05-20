from fastapi.testclient import TestClient

from src.api.main import app
from src.api.conversation_store import ConversationStore
import src.api.routes.conversations as conversations_module
import src.api.routes.chat as chat_module


def test_export_endpoints_return_requested_formats(tmp_path, monkeypatch):
    store = ConversationStore(tmp_path / "exports.db")
    monkeypatch.setenv("API_KEYS", "alpha")
    from src.api.auth import reset_auth_config_cache

    reset_auth_config_cache()
    monkeypatch.setattr(conversations_module, "get_conversation_store", lambda: store)
    monkeypatch.setattr(chat_module, "get_conversation_store", lambda: store)

    store.append_turn(
        conversation_id="conv-1",
        user_query="Was ist empfohlen?",
        final_state={
            "answer_professional": "Fachliche Antwort",
            "answer_plain": "Einfache Antwort",
            "citations": [],
            "retrieved_chunks": [],
            "tool_calls_log": [],
            "rag_trace": [{"name": "rewrite", "status": "ok"}],
            "token_usage": {"total_tokens": 12, "cost_usd": 0.0001, "calls": []},
            "external_search_snippets": [],
        },
        combined_answer="Fachliche Antwort\n\nEinfache Antwort",
    )

    client = TestClient(app)

    json_resp = client.get("/conversations/conv-1/export?format=json", headers={"X-API-Key": "alpha"})
    assert json_resp.status_code == 200
    assert json_resp.headers["content-type"].startswith("application/json")

    csv_resp = client.get("/conversations/conv-1/export?format=csv", headers={"X-API-Key": "alpha"})
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "conversation_id" in csv_resp.text

    pdf_resp = client.get("/conversations/conv-1/export?format=pdf", headers={"X-API-Key": "alpha"})
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"].startswith("application/pdf")
    assert pdf_resp.content.startswith(b"%PDF")