import json
import httpx
import streamlit as st
from src.ui.components.source_cards import render_source_cards, render_tool_calls
from src.ui.components.filters import render_feedback_buttons


def render_chat_page(api_url: str, api_key: str, filters: dict) -> None:
    st.title("Onkologie Leitlinien-Assistent")
    st.caption("S3-Leitlinien: Mammakarzinom · Kolorektales Karzinom · Lungenkarzinom · Prostatakarzinom")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Stellen Sie Ihre Frage zu den Leitlinien...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Suche in den Leitlinien..."):
            try:
                resp = httpx.post(
                    f"{api_url}/chat",
                    json={
                        "query": query,
                        "session_id": st.session_state.session_id,
                        **filters,
                    },
                    headers={"X-API-Key": api_key},
                    timeout=120,
                )
                resp.raise_for_status()

                # Parse SSE
                payload = None
                for line in resp.text.splitlines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        payload = json.loads(line[5:].strip())

                if payload and not payload.get("blocked"):
                    st.markdown("**Fachliche Antwort**")
                    st.markdown(payload["answer_professional"])
                    st.markdown("---")
                    st.markdown("**In einfachen Worten**")
                    st.markdown(payload["answer_plain"])
                    st.markdown(payload.get("disclaimer", ""))
                    render_source_cards(payload.get("citations", []))
                    render_tool_calls(payload.get("tool_calls", []))
                    render_feedback_buttons(
                        st.session_state.session_id, query, api_url, api_key
                    )
                    answer_text = payload["answer_professional"]
                elif payload and payload.get("blocked"):
                    st.warning(payload.get("answer_professional", "Anfrage blockiert."))
                    answer_text = payload.get("answer_professional", "")
                else:
                    st.error("Keine Antwort erhalten.")
                    answer_text = ""

                st.session_state.messages.append({"role": "assistant", "content": answer_text})

            except Exception as e:
                st.error(f"Fehler: {e}")
