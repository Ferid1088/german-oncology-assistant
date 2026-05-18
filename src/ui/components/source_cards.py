import streamlit as st


def render_source_cards(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("#### Quellen")
    for c in citations:
        with st.expander(f"{c['label']} {c['citation']}"):
            st.markdown(f"**Datei:** `{c.get('source_filename', 'n/a')}`")
            st.markdown(f"**Referenz:** {c['citation']}")


def render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"Tool-Aufrufe ({len(tool_calls)})"):
        for tc in tool_calls:
            st.json(tc)
