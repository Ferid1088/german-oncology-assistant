from src.graph.state import RAGState

ONCOLOGY_KEYWORDS = [
    "karzinom", "tumor", "krebs", "leitlinie", "empfehlung", "therapie",
    "diagnose", "screening", "onkologie", "chemo", "bestrahlung", "mamma",
    "prostat", "lunge", "kolorektal", "darm", "metastas", "staging",
    "evidenz", "grade", "studie", "patient",
]


def apply_input_guardrail(state: RAGState) -> dict:
    query = state["user_query"].lower()

    has_medical = any(kw in query for kw in ONCOLOGY_KEYWORDS)
    if not has_medical and len(query.split()) > 3:
        return {
            "input_blocked": True,
            "input_block_reason": "Ihre Anfrage scheint nicht onkologische Leitlinien zu betreffen. Bitte stellen Sie medizinische Fragen zu den S3-Leitlinien.",
        }

    return {"input_blocked": False, "input_block_reason": ""}
