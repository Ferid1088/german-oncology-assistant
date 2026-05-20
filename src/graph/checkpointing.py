from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)


def build_checkpointer():
    """Best-effort PostgresSaver integration path.

    If LangGraph PostgresSaver or DATABASE_URL is unavailable, return None so the
    graph still runs without persistence.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore

        return PostgresSaver.from_conn_string(database_url)
    except Exception as exc:  # pragma: no cover - optional dependency path
        log.warning("PostgresSaver unavailable, continuing without checkpointer: %s", exc)
        return None