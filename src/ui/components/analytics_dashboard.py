from __future__ import annotations

import httpx
import streamlit as st


def _safe_message_count(conversation: dict | None) -> int:
    messages = (conversation or {}).get("messages", [])
    return len(messages) if isinstance(messages, list) else 0


def _safe_assistant_count(conversation: dict | None) -> int:
    messages = (conversation or {}).get("messages", [])
    if not isinstance(messages, list):
        return 0
    return sum(1 for message in messages if message.get("role") == "assistant")


def _active_filter_count(filters: dict | None) -> int:
    payload = filters or {}
    return sum(1 for value in payload.values() if value)


def _format_rate(value: float | int | None) -> str:
    return f"{float(value or 0.0) * 100:.1f}%"


def _load_analytics(api_url: str, api_key: str, session_id: str | None) -> tuple[dict | None, str | None]:
    try:
        response = httpx.get(
            f"{api_url}/analytics/overview",
            params={"session_id": session_id} if session_id else None,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        return response.json(), None
    except httpx.HTTPError as exc:
        return None, str(exc)


def _render_overview(conversation: dict | None, filters: dict | None, analytics: dict | None) -> None:
    overview = (analytics or {}).get("overview", {})
    title = (conversation or {}).get("title") or "Neue Konversation"
    session_id = (conversation or {}).get("session_id") or "—"

    col1, col2, col3 = st.columns(3)
    col1.metric("Conversations", overview.get("total_conversations", 0))
    col2.metric("Questions", overview.get("total_questions", 0))
    col3.metric("Answers", overview.get("total_answers", 0))

    col4, col5, col6 = st.columns(3)
    col4.metric("Tokens", f"{int(overview.get('total_tokens', 0) or 0):,}")
    col5.metric("Cost", f"${float(overview.get('total_cost_usd', 0.0) or 0.0):.4f}")
    col6.metric("Avg tokens/answer", f"{float(overview.get('avg_tokens_per_answer', 0.0) or 0.0):.1f}")

    col7, col8, col9 = st.columns(3)
    col7.metric("Citation coverage", _format_rate(overview.get("citation_coverage_rate")))
    col8.metric("Tool usage rate", _format_rate(overview.get("tool_usage_rate")))
    col9.metric("External search rate", _format_rate(overview.get("external_search_rate")))

    st.caption("Current workspace snapshot")
    st.markdown(f"**Conversation**  \n{title}")
    st.caption(f"Session ID: {session_id}")
    st.caption(f"Active filters: {_active_filter_count(filters)} · Local messages: {_safe_message_count(conversation)}")


def _render_usage(analytics: dict | None) -> None:
    timeseries = (analytics or {}).get("timeseries", {})
    conversations = timeseries.get("conversations", [])
    questions = timeseries.get("questions", [])
    answers = timeseries.get("answers", [])

    st.markdown("#### Activity over time")
    if conversations:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Conversations by day")
            st.dataframe(conversations, hide_index=True, use_container_width=True)
        with col2:
            st.caption("Questions and answers by day")
            merged = []
            question_map = {row["date"]: row["count"] for row in questions}
            answer_map = {row["date"]: row["count"] for row in answers}
            for date in sorted(set(question_map) | set(answer_map)):
                merged.append({"date": date, "questions": question_map.get(date, 0), "answers": answer_map.get(date, 0)})
            st.dataframe(merged, hide_index=True, use_container_width=True)
    else:
        st.info("No conversation analytics are available yet.")


def _render_distribution_block(title: str, rows: list[dict]) -> None:
    st.caption(title)
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No data available yet.")


def _render_diagnostics(conversation: dict | None, filters: dict | None, analytics: dict | None) -> None:
    distributions = (analytics or {}).get("distributions", {})
    tables = (analytics or {}).get("tables", {})
    current_session = (analytics or {}).get("current_session") or {}

    col1, col2 = st.columns(2)
    with col1:
        _render_distribution_block("Tool usage", distributions.get("tools", []))
        _render_distribution_block("Guideline references", distributions.get("guidelines", []))
    with col2:
        _render_distribution_block("RAG step status", distributions.get("rag_status", []))
        _render_distribution_block("Top sessions by cost", tables.get("top_sessions", []))

    st.caption("Context for the current workspace")
    st.json(
        {
            "session_id": (conversation or {}).get("session_id"),
            "title": (conversation or {}).get("title"),
            "message_count": _safe_message_count(conversation),
            "assistant_turns": _safe_assistant_count(conversation),
            "active_filters": filters or {},
            "current_session_analytics": current_session,
        }
    )


def render_analytics_dashboard(
    api_url: str,
    api_key: str,
    conversation: dict | None = None,
    filters: dict | None = None,
) -> None:
    analytics, error = _load_analytics(api_url, api_key, (conversation or {}).get("session_id"))

    with st.container(border=True):
        st.markdown("### 📊 Analytics")
        st.caption("Advanced dashboard shell attached to the main chat workspace.")
        if error:
            st.warning(f"Analytics service unavailable: {error}")

        overview_tab, usage_tab, diagnostics_tab = st.tabs(["Overview", "Usage", "Diagnostics"])

        with overview_tab:
            _render_overview(conversation, filters, analytics)

        with usage_tab:
            _render_usage(analytics)

        with diagnostics_tab:
            _render_diagnostics(conversation, filters, analytics)