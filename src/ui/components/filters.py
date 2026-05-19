import streamlit as st


def render_filters() -> dict:
    """Render sidebar filter panel. Returns dict of active filters."""
    st.sidebar.header("Filter")
    guideline = st.sidebar.selectbox(
        "Leitlinie",
        options=["Alle", "mamma", "krk", "lunge", "prosta"],
        format_func=lambda x: {
            "Alle": "Alle Leitlinien",
            "mamma": "Mammakarzinom",
            "krk": "Kolorektales Karzinom",
            "lunge": "Lungenkarzinom",
            "prosta": "Prostatakarzinom",
        }.get(x, x),
    )
    grade = st.sidebar.selectbox(
        "Empfehlungsgrad",
        options=["Alle", "A", "B", "0"],
        help=(
            "Filtert Ergebnisse nach dem Empfehlungsgrad der S3-Leitlinien (AWMF):\n\n"
            "**A – Soll:** Starke Empfehlung. Hohe Evidenzqualität, klarer Nutzen. "
            "Soll routinemäßig angewendet werden.\n\n"
            "**B – Sollte:** Moderate Empfehlung. Mittlere Evidenz oder geringerer Konsens. "
            "In den meisten Fällen empfohlen.\n\n"
            "**0 – Kann:** Offene Empfehlung. Schwache Evidenz oder gleichwertige Alternativen. "
            "Liegt im Ermessen des Arztes."
        ),
    )
    return {
        "guideline_id": "" if guideline == "Alle" else guideline,
        "grade": "" if grade == "Alle" else grade,
    }


def render_feedback_buttons(session_id: str, query: str, api_url: str, api_key: str) -> None:
    col1, col2 = st.columns([1, 1])
    import httpx
    with col1:
        if st.button("👍 Hilfreich"):
            httpx.post(f"{api_url}/feedback", json={
                "session_id": session_id, "query": query, "rating": 1
            }, headers={"X-API-Key": api_key})
    with col2:
        if st.button("👎 Nicht hilfreich"):
            httpx.post(f"{api_url}/feedback", json={
                "session_id": session_id, "query": query, "rating": -1
            }, headers={"X-API-Key": api_key})
