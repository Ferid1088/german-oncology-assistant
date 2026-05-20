from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import src.api.routes.chat as chat_module
from src.api.auth import reset_auth_config_cache
from src.api.conversation_store import ConversationStore
from src.api.main import app as main_app
from src.api.observability import configure_observability
from src.api.rate_limit import _RATE_LIMITER


def test_observability_http_exception_returns_trace_id_and_technical_details():
    app = FastAPI()
    configure_observability(app)

    @app.get("/boom")
    async def boom():
        raise HTTPException(status_code=404, detail="missing")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")

    assert response.status_code == 404
    payload = response.json()
    assert payload["trace_id"]
    assert payload["technical_details"]["status_code"] == 404
    assert payload["technical_details"]["path"] == "/boom"
    assert response.headers["X-Trace-Id"] == payload["trace_id"]


def test_observability_unhandled_exception_returns_trace_id_and_technical_details():
    app = FastAPI()
    configure_observability(app)

    @app.get("/explode")
    async def explode():
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/explode")

    assert response.status_code == 500
    payload = response.json()
    assert payload["trace_id"]
    assert payload["technical_details"]["error_type"] == "RuntimeError"
    assert response.headers["X-Trace-Id"] == payload["trace_id"]


def test_chat_error_payload_includes_trace_id_and_technical_details(tmp_path, monkeypatch):
    class BrokenGraph:
        checkpointer = None

        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("graph failure")

    monkeypatch.setenv("API_KEYS", "alpha")
    reset_auth_config_cache()
    _RATE_LIMITER._events.clear()
    monkeypatch.setattr(chat_module, "get_graph", lambda: BrokenGraph())
    monkeypatch.setattr(chat_module, "get_conversation_store", lambda: ConversationStore(tmp_path / "chat-errors.db"))

    client = TestClient(main_app, raise_server_exceptions=False)
    response = client.post(
        "/chat",
        json={"query": "test", "session_id": "s1"},
        headers={"X-API-Key": "alpha"},
    )

    assert response.status_code == 200
    assert response.headers["X-Trace-Id"]

    payload = None
    for line in response.text.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            payload = json.loads(line[5:].strip())

    assert payload is not None
    assert payload["trace_id"] == response.headers["X-Trace-Id"]
    assert payload["technical_details"]["error_type"] == "RuntimeError"
    assert payload["technical_details"]["trace_id"] == payload["trace_id"]