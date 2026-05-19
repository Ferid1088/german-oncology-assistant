from src.graph.state import RAGState

BLOCK_KEYWORDS = [
    "wetter", "weather", "sport", "fußball", "soccer", "football", "aktie",
    "stock", "rezept", "recipe", "kochen", "cooking", "politik", "politics",
    "musik", "music", "film", "movie", "reise", "travel",
]


def apply_input_guardrail(state: RAGState) -> dict:
    query = state["user_query"].lower()

    # Only block queries that are clearly unrelated to medicine/oncology
    is_off_topic = any(kw in query for kw in BLOCK_KEYWORDS)
    if is_off_topic:
        return {
            "input_blocked": True,
            "input_block_reason": "Ihre Anfrage scheint nicht onkologische Leitlinien zu betreffen. Bitte stellen Sie medizinische Fragen zu den S3-Leitlinien.",
        }

    return {"input_blocked": False, "input_block_reason": ""}
