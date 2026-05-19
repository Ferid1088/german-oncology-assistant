import json
import uuid
import datetime
import httpx
import streamlit as st
from src.ui.components.source_cards import render_source_cards
from src.ui.components.inline_citations import annotate_citations
from src.ui.components.filters import render_filters, render_feedback_buttons

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


def _date_group(dt: datetime.datetime) -> str:
    delta = (datetime.date.today() - dt.date()).days
    if delta == 0:
        return "Heute"
    if delta == 1:
        return "Gestern"
    if delta < 7:
        return "Letzte 7 Tage"
    if delta < 30:
        return "Letzte 30 Tage"
    return "Älter"


def _new_conversation() -> str:
    new_id = str(uuid.uuid4())
    st.session_state.conversations[new_id] = {
        "session_id": new_id,
        "title": "Neue Konversation",
        "messages": [],
        "created_at": datetime.datetime.now(),
    }
    return new_id


def _delete_conversation(cid: str) -> None:
    del st.session_state.conversations[cid]
    if st.session_state.active_id == cid:
        remaining = list(st.session_state.conversations.keys())
        st.session_state.active_id = remaining[-1] if remaining else _new_conversation()


def _init_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
        st.session_state.active_id = _new_conversation()


def _render_sidebar() -> dict:
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # ── Filters ───────────────────────────────────────────
        filters = render_filters()

        st.divider()

        # ── New Chat ──────────────────────────────────────────
        if st.button("✏️  Neuer Chat", use_container_width=True):
            st.session_state.active_id = _new_conversation()
            st.rerun()

        st.divider()

        # ── Conversation history ───────────────────────────────
        convs = list(reversed(list(st.session_state.conversations.items())))
        current_group = None

        for cid, conv in convs:
            group = _date_group(conv["created_at"])
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
                        _delete_conversation(cid)
                        st.rerun()

    return filters


def render_chat_page(api_url: str, api_key: str) -> None:
    _init_state()
    filters = _render_sidebar()

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
                    pro = annotate_citations(payload["answer_professional"], citations)
                    plain = annotate_citations(payload["answer_plain"], citations)
                    st.markdown("**Fachliche Antwort**")
                    st.markdown(pro, unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("**In einfachen Worten**")
                    st.markdown(plain, unsafe_allow_html=True)
                    st.markdown(payload.get("disclaimer", ""))
                    render_source_cards(citations)
                    render_feedback_buttons(conv["session_id"], query, api_url, api_key)
                    answer_text = payload["answer_professional"]
                elif payload and payload.get("blocked"):
                    st.warning(payload.get("answer_professional", "Anfrage blockiert."))
                    answer_text = payload.get("answer_professional", "")
                else:
                    st.error("Keine Antwort erhalten.")
                    answer_text = ""

                conv["messages"].append({"role": "assistant", "content": answer_text})

            except Exception as e:
                st.error(f"Fehler: {e}")
