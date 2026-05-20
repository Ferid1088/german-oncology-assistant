from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from fastapi import HTTPException, Request


CONFIG_PATH = Path(__file__).with_name("rate_limit.config.json")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@dataclass(frozen=True)
class RateLimitPolicy:
    limit: int
    window_seconds: int


@dataclass(frozen=True)
class RateLimitConfig:
    bucket_key_parts: tuple[str, ...]
    route_groups: dict[str, RateLimitPolicy]
    route_patterns: tuple[tuple[re.Pattern[str], str], ...]


class RateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, bucket: str, policy: RateLimitPolicy) -> int | None:
        now = time.time()
        cutoff = now - policy.window_seconds

        with self._lock:
            events = self._events.setdefault(bucket, deque())
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= policy.limit:
                retry_after = max(1, int(events[0] + policy.window_seconds - now + 0.999))
                return retry_after

            events.append(now)
            if not events:
                self._events.pop(bucket, None)
            return None


_RATE_LIMITER = RateLimiter()


def _load_config() -> RateLimitConfig:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if "bucket_key_parts" not in payload:
        raise ValueError("rate_limit.config.json must define 'bucket_key_parts'")
    if "route_groups" not in payload:
        raise ValueError("rate_limit.config.json must define 'route_groups'")

    bucket_key_parts = tuple(payload["bucket_key_parts"])
    route_groups_payload = payload["route_groups"]

    route_groups: dict[str, RateLimitPolicy] = {}
    route_patterns: list[tuple[re.Pattern[str], str]] = []
    for group_name, group_payload in route_groups_payload.items():
        route_groups[group_name] = RateLimitPolicy(
            limit=max(1, int(group_payload["limit"])),
            window_seconds=max(1, int(group_payload["window_seconds"])),
        )
        for route in group_payload.get("routes", []):
            route_regex = "^" + re.sub(r"\{[^/]+\}", r"[^/]+", route) + "$"
            route_patterns.append((re.compile(route_regex), group_name))

    return RateLimitConfig(
        bucket_key_parts=bucket_key_parts,
        route_groups=route_groups,
        route_patterns=tuple(route_patterns),
    )


_CONFIG = _load_config()


def reset_rate_limit_config() -> None:
    global _CONFIG
    _CONFIG = _load_config()


def _policy_for(route_group: str) -> RateLimitPolicy:
    policy = _CONFIG.route_groups.get(route_group)
    if policy is None:
        raise ValueError(f"Unknown route group: {route_group}")
    return policy


def route_group_for_path(path: str) -> str:
    for pattern, group in _CONFIG.route_patterns:
        if pattern.match(path):
            return group

    raise ValueError(f"No rate-limit route group configured for path: {path}")


def _bucket_value(part: str, *, route_group: str, api_key: str, ip: str) -> str:
    values = {
        "route_group": route_group,
        "api_key": api_key,
        "ip": ip,
    }
    value = values.get(part)
    if value is None:
        raise ValueError(f"Unsupported bucket key part: {part}")
    return value


def _bucket_key(route_group: str, api_key: str, ip: str) -> str:
    return ":".join(
        _bucket_value(part, route_group=route_group, api_key=api_key, ip=ip)
        for part in _CONFIG.bucket_key_parts
    )


def enforce_rate_limit(request: Request, api_key: str, route_group: str) -> None:
    policy = _policy_for(route_group)
    ip = _client_ip(request)
    bucket = _bucket_key(route_group, api_key, ip)
    retry_after = _RATE_LIMITER.check(bucket, policy)
    if retry_after is None:
        return

    raise HTTPException(
        status_code=429,
        detail={
            "message": "Too many requests",
            "reason": (
                "You have reached usage protection limits to prevent abuse and manage system cost."
            ),
            "retry_after_seconds": retry_after,
            "route_group": route_group,
        },
        headers={"Retry-After": str(retry_after)},
    )
