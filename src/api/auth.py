from __future__ import annotations

import os
from functools import lru_cache

from fastapi import HTTPException, Request


@lru_cache(maxsize=1)
def _load_api_keys() -> set[str]:
    keys: set[str] = set()

    multi = os.getenv("API_KEYS", "")
    if multi.strip():
        keys.update(part.strip() for part in multi.split(",") if part.strip())

    single = os.getenv("API_KEY", "")
    if single.strip():
        keys.add(single.strip())

    if not keys and os.getenv("ENV", "development").lower() != "production":
        keys.add("dev-secret-key")

    return keys


def reset_auth_config_cache() -> None:
    _load_api_keys.cache_clear()


def verify_api_key(request: Request) -> str:
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    valid_keys = _load_api_keys()
    if not key or key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key
