from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi import Request
from fastapi.testclient import TestClient

from src.api.auth import reset_auth_config_cache, verify_api_key
import src.api.rate_limit as rate_limit_module
from src.api.rate_limit import _RATE_LIMITER, enforce_rate_limit, route_group_for_path


def _write_rate_limit_config(tmp_path, payload: dict) -> None:
    rate_limit_module.CONFIG_PATH.write_text(json.dumps(payload), encoding="utf-8")
    rate_limit_module.reset_rate_limit_config()


def test_verify_api_key_accepts_configured_keys(monkeypatch):
    monkeypatch.setenv("API_KEYS", "alpha,beta")
    reset_auth_config_cache()

    app = FastAPI()

    @app.get("/protected")
    async def protected(request: Request):
        verify_api_key(request)
        return {"status": "ok"}

    client = TestClient(app)

    ok = client.get("/protected", headers={"X-API-Key": "alpha"})
    assert ok.status_code == 200

    denied = client.get("/protected", headers={"X-API-Key": "wrong"})
    assert denied.status_code == 401


def test_rate_limit_uses_api_key_and_ip(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEYS", "alpha")
    reset_auth_config_cache()
    _RATE_LIMITER._events.clear()
    original = rate_limit_module.CONFIG_PATH.read_text(encoding="utf-8")
    config = {
        "bucket_key_parts": ["route_group", "api_key", "ip"],
        "route_groups": {
            "chat": {"limit": 1, "window_seconds": 60, "routes": ["/limited"]}
        },
    }
    _write_rate_limit_config(tmp_path, config)

    app = FastAPI()

    @app.get("/limited")
    async def limited(request: Request):
        api_key = verify_api_key(request)
        enforce_rate_limit(request, api_key=api_key, route_group=route_group_for_path(request.url.path))
        return {"status": "ok"}

    client = TestClient(app)

    first = client.get("/limited", headers={"X-API-Key": "alpha", "X-Forwarded-For": "1.2.3.4"})
    assert first.status_code == 200

    second = client.get("/limited", headers={"X-API-Key": "alpha", "X-Forwarded-For": "1.2.3.4"})
    assert second.status_code == 429
    assert second.json()["detail"]["retry_after_seconds"] >= 1

    different_ip = client.get("/limited", headers={"X-API-Key": "alpha", "X-Forwarded-For": "5.6.7.8"})
    assert different_ip.status_code == 200
    rate_limit_module.CONFIG_PATH.write_text(original, encoding="utf-8")
    rate_limit_module.reset_rate_limit_config()


def test_rate_limit_separates_different_api_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEYS", "alpha,beta")
    reset_auth_config_cache()
    _RATE_LIMITER._events.clear()
    original = rate_limit_module.CONFIG_PATH.read_text(encoding="utf-8")
    config = {
        "bucket_key_parts": ["route_group", "api_key", "ip"],
        "route_groups": {
            "chat": {"limit": 1, "window_seconds": 60, "routes": ["/limited"]}
        },
    }
    _write_rate_limit_config(tmp_path, config)

    app = FastAPI()

    @app.get("/limited")
    async def limited(request: Request):
        api_key = verify_api_key(request)
        enforce_rate_limit(request, api_key=api_key, route_group=route_group_for_path(request.url.path))
        return {"status": "ok"}

    client = TestClient(app)

    first_alpha = client.get("/limited", headers={"X-API-Key": "alpha", "X-Forwarded-For": "1.2.3.4"})
    assert first_alpha.status_code == 200

    second_alpha = client.get("/limited", headers={"X-API-Key": "alpha", "X-Forwarded-For": "1.2.3.4"})
    assert second_alpha.status_code == 429

    first_beta = client.get("/limited", headers={"X-API-Key": "beta", "X-Forwarded-For": "1.2.3.4"})
    assert first_beta.status_code == 200
    rate_limit_module.CONFIG_PATH.write_text(original, encoding="utf-8")
    rate_limit_module.reset_rate_limit_config()


def test_route_group_is_loaded_from_json_config(tmp_path):
    original = rate_limit_module.CONFIG_PATH.read_text(encoding="utf-8")
    config = {
        "bucket_key_parts": ["route_group", "api_key", "ip"],
        "route_groups": {
            "chat": {"limit": 7, "window_seconds": 30, "routes": ["/chat"]},
            "general": {"limit": 11, "window_seconds": 45, "routes": ["/conversations", "/conversations/{session_id}", "/conversations/{session_id}/export"]},
            "feedback": {"limit": 13, "window_seconds": 90, "routes": ["/feedback"]}
        }
    }
    _write_rate_limit_config(tmp_path, config)

    assert route_group_for_path("/chat") == "chat"
    assert route_group_for_path("/feedback") == "feedback"
    assert route_group_for_path("/conversations") == "general"
    assert route_group_for_path("/conversations/abc") == "general"
    assert route_group_for_path("/conversations/abc/export") == "general"

    policy = rate_limit_module._policy_for("chat")
    assert policy.limit == 7
    assert policy.window_seconds == 30

    rate_limit_module.CONFIG_PATH.write_text(original, encoding="utf-8")
    rate_limit_module.reset_rate_limit_config()

