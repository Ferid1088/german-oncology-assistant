import json
import uuid
import datetime
from urllib.parse import quote
import httpx
import streamlit as st
from src.ui.components.source_cards import render_source_cards, render_tool_calls
from src.ui.components.inline_citations import annotate_citations
from src.ui.components.filters import render_filters, render_feedback_buttons
from src.ui.components.insights_panels import (
    render_external_search_panel,
    render_rag_process_panel,
    render_token_usage_panel,
)

_CSS = """
<style>
/* ── Sidebar dark background ── */
[data-testid="stSidebar"] {
    background-color: #202123 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #202123 !important;
    padding-top: 12px;
}

/* ── All sidebar text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #ececec !important;
}

/* ── Conversation + New Chat buttons: plain text rows ── */
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    color: #ececec !important;
    text-align: left !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    padding: 7px 10px !important;
    transition: background 0.15s !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
}

/* ── Active conversation (primary type) ── */
[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
    background-color: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
    background-color: rgba(255,255,255,0.16) !important;
}


/* ── "⋯" popover — remove ALL white boxes ── */
[data-testid="stSidebar"] [data-testid="stPopover"],
[data-testid="stSidebar"] [data-testid="stPopover"] > div,
[data-testid="stSidebar"] [data-testid="stPopover"] > * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 4px !important;
    color: #8e8ea0 !important;
    font-size: 18px !important;
    padding: 2px 5px !important;
    line-height: 1 !important;
    min-height: unset !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #ececec !important;
}
/* ── Column containers inside sidebar: transparent ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
[data-testid="stSidebar"] [data-testid="column"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    gap: 2px !important;
    padding: 0 !important;
}

/* ── Dividers ── */
[data-testid="stSidebar"] hr {
    border-color: #3a3a3a !important;
    margin: 6px 0 !important;
}

/* ── Filter selectboxes dark theme ── */
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background-color: #2a2b32 !important;
    border-color: #565869 !important;
    color: #ececec !important;
}
[data-testid="stSidebar"] .stSelectbox svg,
[data-testid="stSidebar"] .stTooltipIcon svg {
    fill: #8e8ea0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #ececec !important;
}

/* ── Filter / section headers ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #8e8ea0 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Inline citation hover tooltips ── */
.cit-wrap {
    position: relative;
    display: inline-block;
    vertical-align: super;
    font-size: 0.75em;
    line-height: 1;
}
.cit-num {
    color: #4a9eff;
    font-weight: 700;
    cursor: help;
    background: rgba(74, 158, 255, 0.12);
    border-radius: 3px;
    padding: 1px 4px;
    transition: background 0.15s;
}
.cit-wrap:hover .cit-num {
    background: rgba(74, 158, 255, 0.28);
}
.cit-tip {
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: #1e1f26;
    color: #ececec;
    border: 1px solid #3a3b47;
    border-radius: 10px;
    padding: 10px 14px;
    min-width: 240px;
    max-width: 340px;
    font-size: 13px;
    font-weight: normal;
    line-height: 1.6;
    white-space: normal;
    z-index: 9999;
    box-shadow: 0 8px 28px rgba(0,0,0,0.65);
    vertical-align: baseline;
    pointer-events: none;
}
.cit-tip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 7px solid transparent;
    border-top-color: #3a3b47;
}
.cit-wrap:hover .cit-tip {
    display: block;
}
</style>
"""

_DATE_LABEL_STYLE = (
    "color:#8e8ea0;font-size:11px;font-weight:600;"
    "padding:10px 10px 3px;text-transform:uppercase;"
    "letter-spacing:.05em;margin:0;line-height:1.4"
)


def _combine_answer_parts(answer_professional: str, answer_plain: str) -> str:
    if not answer_plain:
        return (answer_professional or "").strip()

    parts = []
    if answer_professional:
        parts.append(f"Fachliche Antwort\n\n{answer_professional}")
    if answer_plain:
        parts.append(f"In einfachen Worten\n\n{answer_plain}")
    return "\n\n".join(parts).strip()


def _is_clarification_payload(payload: dict) -> bool:
    return bool(payload.get("requires_clarification")) and not payload.get("answer_plain")


def _as_datetime(value) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.datetime.now(datetime.timezone.utc)


def _date_group(dt: datetime.datetime | str) -> str:
    dt_value = _as_datetime(dt)
    delta = (datetime.date.today() - dt_value.date()).days
    if delta == 0:
        return "Heute"
    if delta == 1:
        return "Gestern"
    if delta < 7:
        return "Letzte 7 Tage"
    if delta < 30:
        return "Letzte 30 Tage"
    return "Älter"


def _api_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}


def _load_conversations(api_url: str, api_key: str) -> dict[str, dict]:
    response = httpx.get(
        f"{api_url}/conversations",
        headers=_api_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    conversations = payload.get("conversations", [])
    return {conversation["session_id"]: conversation for conversation in conversations}


def _show_backend_unavailable(api_url: str, error: Exception) -> None:
    st.title("Onkologie Leitlinien-Assistent")
    st.error(
        "Die UI konnte den Backend-Service noch nicht erreichen. "
        "Bitte warten Sie einen Moment und versuchen Sie es erneut."
    )
    st.caption(f"Backend URL: {api_url}")
    with st.expander("Technical details", expanded=False):
        st.code(str(error))
    if st.button("Erneut versuchen", use_container_width=True):
        st.rerun()


def _render_validation_feedback(payload: dict) -> None:
    technical = payload.get("technical_details") or {}
    if payload.get("blocked_reason") != "validation" and technical.get("status_code") != 422:
        return

    st.warning(payload.get("answer_professional") or payload.get("message") or "The request is invalid.")
    detail_title = payload.get("technical_title") or "Technical details"
    with st.expander(detail_title, expanded=False):
        st.json(technical)


def _render_request_diagnostics(payload: dict) -> None:
    diagnostics = {
        "trace_id": payload.get("trace_id"),
        "followup_routing": payload.get("followup_routing"),
        "requires_clarification": payload.get("requires_clarification"),
        "missing_clinical_dimensions": payload.get("missing_clinical_dimensions", []),
        "tool_call_count": len(payload.get("tool_calls", [])),
        "safety_warning": payload.get("safety_warning"),
    }
    technical_details = payload.get("technical_details")
    if technical_details:
        diagnostics["technical_details"] = technical_details

    diagnostics = {k: v for k, v in diagnostics.items() if v not in (None, "", [], {})}
    if not diagnostics:
        return

    with st.expander("Request diagnostics", expanded=False):
        st.json(diagnostics)


def _render_safety_panel(payload: dict) -> None:
    warning = payload.get("safety_warning")
    explanation = payload.get("safety_explanation")
    title = payload.get("safety_title") or "Why was this limited?"

    if warning:
        st.warning(warning)
    if explanation:
        with st.expander(title, expanded=False):
            st.markdown(explanation)


def _create_conversation(api_url: str, api_key: str) -> str:
    new_id = str(uuid.uuid4())
    response = httpx.post(
        f"{api_url}/conversations",
        json={"session_id": new_id, "title": "Neue Konversation"},
        headers=_api_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    st.session_state.conversations[new_id] = response.json()
    return new_id


def _delete_conversation(cid: str, api_url: str, api_key: str) -> None:
    response = httpx.delete(
        f"{api_url}/conversations/{cid}",
        headers=_api_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()
    del st.session_state.conversations[cid]
    if st.session_state.active_id == cid:
        remaining = list(st.session_state.conversations.keys())
        st.session_state.active_id = remaining[0] if remaining else None


def _sync_conversations(api_url: str, api_key: str) -> bool:
    try:
        conversations = _load_conversations(api_url, api_key)
    except httpx.HTTPError as exc:
        st.session_state.backend_available = False
        st.session_state.backend_error = str(exc)
        return False

    st.session_state.backend_available = True
    st.session_state.backend_error = ""
    st.session_state.conversations = conversations

    if not conversations:
        st.session_state.active_id = _create_conversation(api_url, api_key)
        return True

    if st.session_state.get("active_id") not in conversations:
        sorted_ids = [
            cid for cid, _conv in sorted(
                conversations.items(),
                key=lambda item: _as_datetime(item[1].get("updated_at") or item[1].get("created_at")),
                reverse=True,
            )
        ]
        st.session_state.active_id = sorted_ids[0]
    return True


def _init_state(api_url: str, api_key: str) -> bool:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "active_id" not in st.session_state:
        st.session_state.active_id = None
    if "backend_available" not in st.session_state:
        st.session_state.backend_available = True
    if "backend_error" not in st.session_state:
        st.session_state.backend_error = ""
    return _sync_conversations(api_url, api_key)


def _render_sidebar(api_url: str, api_key: str) -> dict:
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # ── Filters ───────────────────────────────────────────
        filters = render_filters()

        st.divider()

        # ── New Chat ──────────────────────────────────────────
        if st.button("✏️  Neuer Chat", use_container_width=True):
            st.session_state.active_id = _create_conversation(api_url, api_key)
            st.rerun()

        st.divider()

        # ── Conversation history ───────────────────────────────
        convs = sorted(
            st.session_state.conversations.items(),
            key=lambda item: _as_datetime(item[1].get("updated_at") or item[1].get("created_at")),
            reverse=True,
        )
        current_group = None

        for cid, conv in convs:
            group = _date_group(conv.get("updated_at") or conv.get("created_at"))
            if group != current_group:
                st.markdown(
                    f'<p style="{_DATE_LABEL_STYLE}">{group}</p>',
                    unsafe_allow_html=True,
                )
                current_group = group

            is_active = cid == st.session_state.active_id
            btn_type = "primary" if is_active else "secondary"

            col_title, col_dots = st.columns([11, 1], gap="small")
            with col_title:
                if st.button(
                    conv["title"],
                    key=f"conv_{cid}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    if not is_active:
                        st.session_state.active_id = cid
                        st.rerun()
            with col_dots:
                with st.popover("⋯", use_container_width=True):
                    if st.button("🗑 Löschen", key=f"del_{cid}", use_container_width=True):
                        _delete_conversation(cid, api_url, api_key)
                        st.rerun()

        if st.session_state.get("active_id"):
            session_id = st.session_state.active_id
            encoded_key = quote(api_key, safe="")
            st.divider()
            st.caption("Export conversation")
            st.link_button(
                "JSON",
                f"{api_url}/conversations/{session_id}/export?format=json&api_key={encoded_key}",
                use_container_width=True,
            )
            st.link_button(
                "CSV",
                f"{api_url}/conversations/{session_id}/export?format=csv&api_key={encoded_key}",
                use_container_width=True,
            )
            st.link_button(
                "PDF",
                f"{api_url}/conversations/{session_id}/export?format=pdf&api_key={encoded_key}",
                use_container_width=True,
            )

    return filters


def render_chat_page(api_url: str, api_key: str) -> None:
    if not _init_state(api_url, api_key):
        _show_backend_unavailable(api_url, st.session_state.get("backend_error", "Unbekannter Fehler"))
        return

    filters = _render_sidebar(api_url, api_key)

    conv = st.session_state.conversations[st.session_state.active_id]

    st.title("Onkologie Leitlinien-Assistent")
    st.caption("S3-Leitlinien: Mammakarzinom · Kolorektales Karzinom · Lungenkarzinom · Prostatakarzinom")

    for msg in conv["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Stellen Sie Ihre Frage zu den Leitlinien...")
    if not query:
        return

    if not conv["messages"]:
        conv["title"] = query[:40] + ("..." if len(query) > 40 else "")

    conv["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Suche in den Leitlinien..."):
            try:
                resp = httpx.post(
                    f"{api_url}/chat",
                    json={"query": query, "session_id": conv["session_id"], **filters},
                    headers={"X-API-Key": api_key},
                    timeout=120,
                )
                resp.raise_for_status()

                payload = None
                for line in resp.text.splitlines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        payload = json.loads(line[5:].strip())

                if payload and not payload.get("blocked"):
                    citations = payload.get("citations", [])
                    tool_calls = payload.get("tool_calls", [])
                    rag_trace = payload.get("rag_trace", [])
                    token_usage = payload.get("token_usage", {})
                    external_search_snippets = payload.get("external_search_snippets", [])
                    pro_raw = payload.get("answer_professional", "")
                    plain_raw = payload.get("answer_plain", "")
                    pro = annotate_citations(pro_raw, citations) if pro_raw else ""
                    plain = annotate_citations(plain_raw, citations) if plain_raw else ""
                    clarification_only = _is_clarification_payload(payload)
                    main_col, side_col = st.columns([3, 1], gap="large")
                    with main_col:
                        _render_safety_panel(payload)
                        if clarification_only and pro:
                            st.markdown(pro, unsafe_allow_html=True)
                        elif pro:
                            st.markdown("**Fachliche Antwort**")
                            st.markdown(pro, unsafe_allow_html=True)
                        if plain:
                            if pro:
                                st.markdown("---")
                            st.markdown("**In einfachen Worten**")
                            st.markdown(plain, unsafe_allow_html=True)
                        st.markdown(payload.get("disclaimer", ""))
                        render_source_cards(citations)
                        _render_request_diagnostics(payload)
                        render_feedback_buttons(conv["session_id"], query, api_url, api_key)
                    with side_col:
                        render_token_usage_panel(token_usage)
                        render_rag_process_panel(rag_trace)
                        if tool_calls:
                            render_tool_calls(tool_calls)
                        render_external_search_panel(external_search_snippets)
                    answer_text = _combine_answer_parts(pro_raw, plain_raw)
                elif payload and payload.get("blocked"):
                    _render_validation_feedback(payload)
                    _render_safety_panel(payload)
                    st.warning(payload.get("answer_professional", "Anfrage blockiert."))
                    plain_raw = payload.get("answer_plain", "")
                    if plain_raw:
                        st.markdown(plain_raw)
                    retry_after = payload.get("retry_after_seconds")
                    if retry_after:
                        st.caption(f"Retry in {retry_after} seconds.")
                    explanation_title = payload.get("blocked_explanation_title") or "Why?"
                    explanation = payload.get("blocked_explanation")
                    if explanation:
                        with st.expander(explanation_title, expanded=False):
                            st.markdown(explanation)
                    render_rag_process_panel(payload.get("rag_trace", []))
                    _render_request_diagnostics(payload)
                    answer_text = _combine_answer_parts(payload.get("answer_professional", ""), plain_raw)
                else:
                    st.error("Keine Antwort erhalten.")
                    answer_text = ""

                conv["messages"].append({"role": "assistant", "content": answer_text})

            except httpx.HTTPStatusError as e:
                response = e.response
                detail = {}
                try:
                    detail = response.json()
                except Exception:
                    detail = {"message": str(e)}

                if response.status_code == 422:
                    payload = {
                        "blocked": True,
                        "blocked_reason": "validation",
                        "answer_professional": detail.get("message", "The request data is invalid."),
                        "answer_plain": "Please adjust the input and try again.",
                        "technical_title": detail.get("technical_title", "Technical details"),
                        "technical_details": detail.get("technical_details", detail),
                        "trace_id": detail.get("trace_id"),
                    }
                    _render_validation_feedback(payload)
                    plain_raw = payload.get("answer_plain", "")
                    if plain_raw:
                        st.markdown(plain_raw)
                    _render_request_diagnostics(payload)
                    conv["messages"].append({
                        "role": "assistant",
                        "content": _combine_answer_parts(payload.get("answer_professional", ""), plain_raw),
                    })
                else:
                    st.error(f"Fehler: {e}")
            except Exception as e:
                st.error(f"Fehler: {e}")
