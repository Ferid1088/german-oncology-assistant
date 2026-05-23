"""API key authentication for all FastAPI endpoints.

Supports two configuration modes:
- **Single key**: set ``API_KEY`` environment variable.
- **Multiple keys**: set ``API_KEYS`` as a comma-separated list.

In development (``ENV != "production"``), falls back to the literal key
``"dev-secret-key"`` when no env vars are configured, so the API works
out of the box without configuration.

Keys are loaded once at startup via ``@lru_cache`` and can be reloaded
(e.g. after a key rotation) by calling ``reset_auth_config_cache()``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import HTTPException, Request


@lru_cache(maxsize=1)
def _load_api_keys() -> set[str]:
    """Load the set of valid API keys from environment variables.

    Reads ``API_KEYS`` (comma-separated) and ``API_KEY`` (single) and combines
    both into one set.  Falls back to ``"dev-secret-key"`` in non-production
    environments when neither variable is set.

    Returns:
        Set of non-empty, stripped API key strings.
    """
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
    """Invalidate the cached key set so it is reloaded on the next request.

    Call after rotating API keys without restarting the server.
    """
    _load_api_keys.cache_clear()


def verify_api_key(request: Request) -> str:
    """FastAPI dependency: validate the API key supplied in the request.

    Accepts the key via the ``X-API-Key`` header or the ``api_key`` query param.

    Args:
        request: Incoming FastAPI request object.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 401 when the key is missing or not in the valid set.
    """
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    valid_keys = _load_api_keys()
    if not key or key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key
