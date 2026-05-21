from __future__ import annotations

import streamlit as st


def _format_cost(cost_usd: float | int | None) -> str:
    value = float(cost_usd or 0.0)
    if value >= 0.01:
        return f"${value:.2f}"
    return f"${value:.4f}"


def render_token_usage_panel(token_usage: dict | None) -> None:
    usage = token_usage or {}
    if not usage:
        return
    total = int(usage.get("total_tokens", 0) or 0)
    if total <= 0 and not usage.get("cost_usd"):
        return

    st.markdown("### Usage")
    col1, col2 = st.columns(2)
    col1.metric("Tokens", f"{total:,}")
    col2.metric("Cost", _format_cost(usage.get("cost_usd", 0.0)))

    with st.expander("Usage details", expanded=False):
        st.caption(
            f"Input: {int(usage.get('input_tokens', 0) or 0):,} · Output: {int(usage.get('output_tokens', 0) or 0):,}"
        )
        calls = usage.get("calls", [])
        if isinstance(calls, list) and calls:
            for call in calls:
                label = call.get("step", "LLM call")
                model = call.get("model", "unknown")
                st.caption(
                    f"• {label}: {call.get('total_tokens', 0)} tokens · {model} · {_format_cost(call.get('cost_usd', 0.0))}"
                )


def render_rag_process_panel(rag_trace: list[dict] | None, *, expand_steps: bool = False) -> None:
    if not rag_trace:
        return
    st.markdown("### RAG process")
    for step in rag_trace:
        status = step.get("status", "ok")
        icon = {
            "ok": "✅",
            "skipped": "⏭️",
            "error": "⚠️",
            "empty": "ℹ️",
            "blocked": "⛔",
        }.get(status, "•")
        duration = step.get("duration_ms")
        extra = f" · {duration:.0f} ms" if isinstance(duration, (int, float)) else ""
        with st.expander(
            f"{icon} {step.get('name', 'step')} — {step.get('summary', '')[:60]}{extra}",
            expanded=expand_steps,
        ):
            st.caption(step.get("summary", ""))
            details = step.get("details", {})
            if details:
                st.json(details)


def render_tool_results_panel(tool_calls: list[dict] | None) -> None:
    if not tool_calls:
        return
    st.markdown("### Tool results")
    for call in tool_calls:
        summary = call.get("summary") or call.get("tool", "tool")
        tool = call.get("tool", "tool")
        status = call.get("status", "ok")
        icon = "✅" if status == "ok" else ("⚠️" if status == "error" else "ℹ️")
        with st.expander(f"{icon} {tool}", expanded=False):
            st.caption(summary)
            preview = call.get("preview", [])
            if isinstance(preview, list):
                for item in preview:
                    st.markdown(f"- {item}")
            if call.get("args"):
                with st.expander("Arguments", expanded=False):
                    st.json(call.get("args"))
            technical = {k: v for k, v in call.items() if k not in {"tool", "summary", "preview", "args"}}
            if technical:
                with st.expander("Technical details", expanded=False):
                    st.json(technical)


def render_external_search_panel(snippets: list[dict] | None) -> None:
    if not snippets:
        return
    st.markdown("### From google search:")
    for snippet in snippets[:5]:
        title = snippet.get("title") or snippet.get("source") or "External result"
        st.markdown(f"**{title}**")
        if snippet.get("snippet"):
            st.caption(snippet.get("snippet"))
        if snippet.get("source") or snippet.get("url"):
            source = snippet.get("source") or snippet.get("url")
            st.caption(f"🔗 {source}")