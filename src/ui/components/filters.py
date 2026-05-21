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
