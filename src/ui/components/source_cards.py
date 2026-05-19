import streamlit as st


def render_source_cards(citations: list[dict]) -> None:
    if not citations:
        return

    has_opinion = any(c.get("is_opinion") for c in citations)
    guideline_count = sum(1 for c in citations if not c.get("is_opinion"))

    if guideline_count > 0 and has_opinion:
        tab_label = f"📎 Quelle(n) ({guideline_count}) · ⚠️ Meine Meinung"
    elif guideline_count > 0:
        tab_label = f"📎 Quelle(n) ({guideline_count})"
    else:
        tab_label = "⚠️ Meine Meinung"

    with st.expander(tab_label, expanded=False):
        for c in citations:
            if c.get("is_opinion"):
                st.warning(
                    "⚠️ **Meine Meinung** — "
                    "Dieser Teil der Antwort basiert auf dem Trainingswissen des Modells "
                    "und ist nicht durch die Leitlinien belegt."
                )
                continue

            st.markdown(f"**{c['label']}**")

            col1, col2 = st.columns([1, 2])
            with col1:
                if c.get("source_filename"):
                    st.caption(f"📄 {c['source_filename']}")
                if c.get("guideline_id"):
                    st.caption(f"🏷 {c['guideline_id'].upper()}")
            with col2:
                if c.get("section_title"):
                    st.caption(f"📑 {c['section_title']}")
                if c.get("section_path"):
                    path_parts = c["section_path"]
                    if isinstance(path_parts, list) and path_parts:
                        st.caption(f"📍 {' › '.join(str(p) for p in path_parts if p)}")
                if c.get("page_start"):
                    page = f"Seite {c['page_start']}"
                    if c.get("page_end") and c["page_end"] != c["page_start"]:
                        page += f"–{c['page_end']}"
                    st.caption(f"🔖 {page}")

            st.divider()


def render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"Tool-Aufrufe ({len(tool_calls)})"):
        for tc in tool_calls:
            st.json(tc)
