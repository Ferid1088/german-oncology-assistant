from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.auth import reset_auth_config_cache
from src.api.main import app
from src.api.rate_limit import _RATE_LIMITER


def test_chat_validation_rejects_invalid_guideline_id(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alpha")
    reset_auth_config_cache()
    _RATE_LIMITER._events.clear()

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/chat",
        json={"query": "Was ist empfohlen?", "session_id": "s1", "guideline_id": "invalid"},
        headers={"X-API-Key": "alpha"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "The request data is invalid."
    assert payload["trace_id"]
    assert payload["technical_details"]["status_code"] == 422


def test_chat_validation_rejects_too_short_query(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alpha")
    reset_auth_config_cache()
    _RATE_LIMITER._events.clear()

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/chat",
        json={"query": "hi", "session_id": "s1"},
        headers={"X-API-Key": "alpha"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "The request data is invalid."
    assert payload["technical_details"]["error_type"] == "RequestValidationError"