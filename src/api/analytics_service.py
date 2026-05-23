"""Analytics aggregation service for the oncology RAG dashboard.

Reads all conversations from ``ConversationStore`` and computes the overview metrics
shown on the analytics dashboard:
- Aggregate counters (total questions, tokens, cost, citation and tool usage rates).
- Time-series data bucketed by UTC date (conversations, questions, answers per day).
- Distribution tables for guidelines cited, tools called, and RAG pipeline step statuses.
- Per-session summaries sorted by cost descending.
- Current-session highlight when ``session_id`` is supplied.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from src.api.conversation_store import ConversationStore


def _date_bucket(value: str | None) -> str:
    """Convert an ISO 8601 timestamp string to a ``YYYY-MM-DD`` date string.

    Args:
        value: ISO 8601 datetime string, or ``None``.

    Returns:
        Date string, or ``"unknown"`` when parsing fails.
    """
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return value[:10]


def _sort_counter(counter: Counter[str], *, limit: int | None = None) -> list[dict]:
    """Sort a Counter by count descending (label ascending as tie-breaker) and format as a list of dicts.

    Args:
        counter: A ``Counter[str]`` of label → count mappings.
        limit: Optional maximum number of items to return.

    Returns:
        List of ``{"label": str, "count": int}`` dicts, highest count first.
    """
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        items = items[:limit]
    return [{"label": label, "count": count} for label, count in items]


def _safe_int(value) -> int:
    """Coerce a value to int, returning 0 on ``None``, empty string, or conversion failure."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    """Coerce a value to float, returning 0.0 on ``None``, empty string, or conversion failure."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ratio(part: int, whole: int) -> float:
    """Return ``part / whole`` rounded to 4 decimal places, or 0.0 when ``whole <= 0``."""
    if whole <= 0:
        return 0.0
    return round(part / whole, 4)


def _session_summary(conversation: dict) -> dict:
    """Compute per-conversation aggregate metrics for the session table.

    Args:
        conversation: Detailed conversation dict including all message payloads.

    Returns:
        A dict with ``session_id``, ``title``, timestamps, turn counts, token/cost
        totals, citation and tool call totals, and ``external_search_turns``.
    """
    messages = conversation.get("messages", []) if isinstance(conversation.get("messages", []), list) else []
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    user_messages = [message for message in messages if message.get("role") == "user"]

    total_tokens = sum(_safe_int((message.get("token_usage") or {}).get("total_tokens")) for message in assistant_messages)
    total_cost = sum(_safe_float((message.get("token_usage") or {}).get("cost_usd")) for message in assistant_messages)
    total_citations = sum(len(message.get("citations", [])) for message in assistant_messages if isinstance(message.get("citations", []), list))
    total_tool_calls = sum(len(message.get("tool_calls", [])) for message in assistant_messages if isinstance(message.get("tool_calls", []), list))
    external_search_turns = sum(1 for message in assistant_messages if message.get("external_search_snippets"))

    return {
        "session_id": conversation.get("session_id"),
        "title": conversation.get("title") or "Neue Konversation",
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "user_turns": len(user_messages),
        "assistant_turns": len(assistant_messages),
        "message_count": len(messages),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "total_citations": total_citations,
        "total_tool_calls": total_tool_calls,
        "external_search_turns": external_search_turns,
    }


def build_analytics_overview(store: ConversationStore, *, session_id: str | None = None) -> dict:
    """Build the full analytics overview payload served by ``GET /analytics/overview``.

    Loads all conversations, iterates every message once, and aggregates all metrics
    in a single pass.  No caching is applied — callers that need caching should wrap
    this function.

    Args:
        store: The ``ConversationStore`` instance to read conversations from.
        session_id: When provided, the matching session is highlighted as
            ``current_session`` in the response payload.

    Returns:
        A dict with ``generated_at``, ``overview``, ``timeseries``, ``distributions``,
        ``tables``, and ``current_session`` keys.
    """
    conversations = store.list_conversations_detailed()

    total_questions = 0
    total_answers = 0
    total_tokens = 0
    total_cost = 0.0
    responses_with_citations = 0
    responses_with_tools = 0
    responses_with_external_search = 0
    total_citations = 0

    conversations_by_day: Counter[str] = Counter()
    questions_by_day: Counter[str] = Counter()
    answers_by_day: Counter[str] = Counter()
    guideline_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    rag_status_counter: Counter[str] = Counter()

    session_summaries: list[dict] = []

    for conversation in conversations:
        conversations_by_day[_date_bucket(conversation.get("created_at"))] += 1
        session_summary = _session_summary(conversation)
        session_summaries.append(session_summary)

        for message in conversation.get("messages", []):
            role = message.get("role")
            if role == "user":
                total_questions += 1
                questions_by_day[_date_bucket(message.get("created_at"))] += 1
                continue

            if role != "assistant":
                continue

            total_answers += 1
            answers_by_day[_date_bucket(message.get("created_at"))] += 1

            token_usage = message.get("token_usage") or {}
            total_tokens += _safe_int(token_usage.get("total_tokens"))
            total_cost += _safe_float(token_usage.get("cost_usd"))

            citations = message.get("citations", []) if isinstance(message.get("citations", []), list) else []
            if citations:
                responses_with_citations += 1
                total_citations += len(citations)
                for citation in citations:
                    guideline_id = citation.get("guideline_id") or "unknown"
                    guideline_counter[str(guideline_id)] += 1

            tool_calls = message.get("tool_calls", []) if isinstance(message.get("tool_calls", []), list) else []
            if tool_calls:
                responses_with_tools += 1
                for tool_call in tool_calls:
                    tool_counter[str(tool_call.get("tool") or "unknown")] += 1

            if message.get("external_search_snippets"):
                responses_with_external_search += 1

            rag_trace = message.get("rag_trace", []) if isinstance(message.get("rag_trace", []), list) else []
            for step in rag_trace:
                rag_status_counter[str(step.get("status") or "unknown")] += 1

    sorted_sessions = sorted(
        session_summaries,
        key=lambda item: (-item["total_cost_usd"], -item["total_tokens"], item["title"]),
    )
    current_session = next((item for item in session_summaries if item.get("session_id") == session_id), None)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": {
            "total_conversations": len(conversations),
            "total_questions": total_questions,
            "total_answers": total_answers,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_tokens_per_answer": round(total_tokens / total_answers, 2) if total_answers else 0.0,
            "avg_questions_per_conversation": round(total_questions / len(conversations), 2) if conversations else 0.0,
            "citation_coverage_rate": _ratio(responses_with_citations, total_answers),
            "tool_usage_rate": _ratio(responses_with_tools, total_answers),
            "external_search_rate": _ratio(responses_with_external_search, total_answers),
            "avg_citations_per_answer": round(total_citations / total_answers, 2) if total_answers else 0.0,
        },
        "timeseries": {
            "conversations": [{"date": date, "count": count} for date, count in sorted(conversations_by_day.items())],
            "questions": [{"date": date, "count": count} for date, count in sorted(questions_by_day.items())],
            "answers": [{"date": date, "count": count} for date, count in sorted(answers_by_day.items())],
        },
        "distributions": {
            "guidelines": _sort_counter(guideline_counter, limit=8),
            "tools": _sort_counter(tool_counter, limit=8),
            "rag_status": _sort_counter(rag_status_counter, limit=8),
        },
        "tables": {
            "top_sessions": sorted_sessions[:5],
        },
        "current_session": current_session,
    }