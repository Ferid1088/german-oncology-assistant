import streamlit as st
from src.citations import format_page_reference


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
                page = format_page_reference(c.get("page_numbers"), c.get("page_start"), c.get("page_end"))
                if page:
                    st.caption(f"🔖 {page}")
                if c.get("recommendation_id"):
                    extra = f"📋 Empfehlung {c['recommendation_id']}"
                    if c.get("recommendation_grade"):
                        extra += f" · Grad {c['recommendation_grade']}"
                    if c.get("evidence_level"):
                        extra += f" · LoE {c['evidence_level']}"
                    st.caption(extra)
                if c.get("reference_ids"):
                    refs = c["reference_ids"]
                    if isinstance(refs, list) and refs:
                        st.caption(f"📚 Referenzen: {', '.join(str(r) for r in refs)}")

            if c.get("contextual_header"):
                st.caption(f"🧭 {c['contextual_header']}")

            if c.get("citation"):
                st.code(c["citation"], language=None)

            st.divider()


def render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return

    st.caption(f"{len(tool_calls)} tool call(s)")
    for tc in tool_calls:
        title = tc.get("tool", "unknown")
        summary = tc.get("summary") or "Tool executed."
        with st.expander(title, expanded=False):
            st.caption(summary)
            preview = tc.get("preview", [])
            if isinstance(preview, list):
                for item in preview:
                    st.markdown(f"- {item}")
            st.json(tc)
