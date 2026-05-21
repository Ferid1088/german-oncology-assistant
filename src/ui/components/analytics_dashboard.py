from __future__ import annotations

import httpx
import streamlit as st


_ANALYTICS_CSS = """
<style>
.analytics-shell {
    background: linear-gradient(180deg, #f8fbff 0%, #f3f7fc 100%);
    border: 1px solid #d9e6f7;
    border-radius: 24px;
    padding: 1rem 1rem 1.1rem 1rem;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}
.analytics-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 58%, #38bdf8 100%);
    color: #ffffff;
    border-radius: 20px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.analytics-eyebrow {
    font-size: 0.76rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.85;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.analytics-title {
    font-size: 1.45rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.analytics-subtitle {
    font-size: 0.95rem;
    line-height: 1.5;
    opacity: 0.92;
}
.analytics-section-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0.35rem 0 0.2rem 0;
}
.analytics-section-copy {
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 0.65rem;
}
.analytics-card {
    border-radius: 18px;
    padding: 0.9rem 0.95rem;
    border: 1px solid #dbe7f6;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    min-height: 118px;
}
.analytics-card-label {
    color: #64748b;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.analytics-card-value {
    color: #0f172a;
    font-size: 1.6rem;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.analytics-card-meta {
    color: #475569;
    font-size: 0.85rem;
    line-height: 1.4;
}
.analytics-card-blue {
    background: linear-gradient(180deg, #ffffff 0%, #f2f7ff 100%);
}
.analytics-card-cyan {
    background: linear-gradient(180deg, #ffffff 0%, #effcff 100%);
}
.analytics-card-violet {
    background: linear-gradient(180deg, #ffffff 0%, #f6f1ff 100%);
}
.analytics-card-amber {
    background: linear-gradient(180deg, #ffffff 0%, #fff8eb 100%);
}
.analytics-card-emerald {
    background: linear-gradient(180deg, #ffffff 0%, #effcf6 100%);
}
.analytics-card-slate {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}
</style>
"""


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


def _metric_card(label: str, value: str, meta: str = "", tone: str = "blue") -> str:
    return f"""
    <div class="analytics-card analytics-card-{tone}">
        <div class="analytics-card-label">{label}</div>
        <div class="analytics-card-value">{value}</div>
        <div class="analytics-card-meta">{meta}</div>
    </div>
    """


def _render_metric_row(items: list[dict]) -> None:
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.markdown(
                _metric_card(
                    item["label"],
                    item["value"],
                    item.get("meta", ""),
                    item.get("tone", "blue"),
                ),
                unsafe_allow_html=True,
            )


def _status_count(rows: list[dict], label: str) -> int:
    for row in rows:
        if row.get("label") == label:
            return int(row.get("count", 0) or 0)
    return 0


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

    st.markdown('<div class="analytics-section-title">Executive snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="analytics-section-copy">A polished overview of assistant usage, cost, and answer quality for the current workspace.</div>',
        unsafe_allow_html=True,
    )

    _render_metric_row(
        [
            {
                "label": "Conversations",
                "value": str(overview.get("total_conversations", 0)),
                "meta": "Tracked sessions in the current analytics window.",
                "tone": "blue",
            },
            {
                "label": "Questions",
                "value": str(overview.get("total_questions", 0)),
                "meta": "User requests processed by the assistant.",
                "tone": "cyan",
            },
            {
                "label": "Answers",
                "value": str(overview.get("total_answers", 0)),
                "meta": "Completed assistant responses available for review.",
                "tone": "violet",
            },
        ]
    )

    _render_metric_row(
        [
            {
                "label": "Total tokens",
                "value": f"{int(overview.get('total_tokens', 0) or 0):,}",
                "meta": f"Avg / answer: {float(overview.get('avg_tokens_per_answer', 0.0) or 0.0):.1f}",
                "tone": "amber",
            },
            {
                "label": "Estimated cost",
                "value": f"${float(overview.get('total_cost_usd', 0.0) or 0.0):.4f}",
                "meta": "Derived from stored per-turn token usage.",
                "tone": "emerald",
            },
            {
                "label": "Citation coverage",
                "value": _format_rate(overview.get("citation_coverage_rate")),
                "meta": f"Avg citations / answer: {float(overview.get('avg_citations_per_answer', 0.0) or 0.0):.1f}",
                "tone": "slate",
            },
        ]
    )

    _render_metric_row(
        [
            {
                "label": "Tool usage",
                "value": _format_rate(overview.get("tool_usage_rate")),
                "meta": "How often an answer required a supporting tool call.",
                "tone": "blue",
            },
            {
                "label": "External search",
                "value": _format_rate(overview.get("external_search_rate")),
                "meta": "Answers enriched with external web results.",
                "tone": "cyan",
            },
            {
                "label": "Workspace filters",
                "value": str(_active_filter_count(filters)),
                "meta": f"Session: {session_id} · Local messages: {_safe_message_count(conversation)}",
                "tone": "violet",
            },
        ]
    )

    with st.expander("Current workspace snapshot", expanded=False):
        st.markdown(f"**Conversation**  \n{title}")
        st.caption(f"Session ID: {session_id}")
        st.caption(f"Assistant turns in this session: {_safe_assistant_count(conversation)}")


def _render_usage(analytics: dict | None) -> None:
    timeseries = (analytics or {}).get("timeseries", {})
    conversations = timeseries.get("conversations", [])
    questions = timeseries.get("questions", [])
    answers = timeseries.get("answers", [])

    st.markdown('<div class="analytics-section-title">Usage trends</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="analytics-section-copy">Click this tab to inspect how conversation volume, question load, and answer activity evolve over time.</div>',
        unsafe_allow_html=True,
    )
    if conversations:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Conversations by day")
            st.dataframe(conversations, hide_index=True, width="stretch")
        with col2:
            st.caption("Questions and answers by day")
            merged = []
            question_map = {row["date"]: row["count"] for row in questions}
            answer_map = {row["date"]: row["count"] for row in answers}
            for date in sorted(set(question_map) | set(answer_map)):
                merged.append({"date": date, "questions": question_map.get(date, 0), "answers": answer_map.get(date, 0)})
            st.dataframe(merged, hide_index=True, width="stretch")
    else:
        st.info("No conversation analytics are available yet.")


def _render_distribution_block(title: str, rows: list[dict]) -> None:
    st.caption(title)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No data available yet.")


def _render_rag_process(analytics: dict | None) -> None:
    distributions = (analytics or {}).get("distributions", {})
    current_session = (analytics or {}).get("current_session") or {}
    rag_status = distributions.get("rag_status", [])

    st.markdown('<div class="analytics-section-title">RAG process monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="analytics-section-copy">Click this tab to review the retrieval-and-generation pipeline performance, step outcomes, and current-session diagnostics.</div>',
        unsafe_allow_html=True,
    )

    _render_metric_row(
        [
            {
                "label": "RAG steps · OK",
                "value": str(_status_count(rag_status, "ok")),
                "meta": "Pipeline steps that completed successfully.",
                "tone": "emerald",
            },
            {
                "label": "RAG steps · Skipped",
                "value": str(_status_count(rag_status, "skipped")),
                "meta": "Steps intentionally bypassed by routing logic.",
                "tone": "amber",
            },
            {
                "label": "RAG steps · Issues",
                "value": str(
                    _status_count(rag_status, "error")
                    + _status_count(rag_status, "blocked")
                    + _status_count(rag_status, "empty")
                ),
                "meta": "Potential weak points worth reviewing.",
                "tone": "violet",
            },
        ]
    )

    _render_metric_row(
        [
            {
                "label": "Current session tools",
                "value": str(current_session.get("total_tool_calls", 0) or 0),
                "meta": "Tool invocations in the selected conversation.",
                "tone": "blue",
            },
            {
                "label": "Current session citations",
                "value": str(current_session.get("total_citations", 0) or 0),
                "meta": "Evidence references attached to this session.",
                "tone": "cyan",
            },
            {
                "label": "External search turns",
                "value": str(current_session.get("external_search_turns", 0) or 0),
                "meta": "Turns that needed search results from outside the core corpus.",
                "tone": "slate",
            },
        ]
    )

    _render_distribution_block("RAG step status distribution", rag_status)


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

    with st.expander("Context for the current workspace", expanded=False):
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

    st.markdown(_ANALYTICS_CSS, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            """
            <div class="analytics-shell">
                <div class="analytics-hero">
                    <div class="analytics-eyebrow">Advanced analytics</div>
                    <div class="analytics-title">Clinical assistant intelligence hub</div>
                    <div class="analytics-subtitle">A cleaner, more visual workspace for usage metrics, RAG process review, and operational diagnostics.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if error:
            st.warning(f"Analytics service unavailable: {error}")

        overview_tab, usage_tab, rag_tab, diagnostics_tab = st.tabs(["Overview", "Usage", "RAG Process", "Diagnostics"])

        with overview_tab:
            _render_overview(conversation, filters, analytics)

        with usage_tab:
            _render_usage(analytics)

        with rag_tab:
            _render_rag_process(analytics)

        with diagnostics_tab:
            _render_diagnostics(conversation, filters, analytics)