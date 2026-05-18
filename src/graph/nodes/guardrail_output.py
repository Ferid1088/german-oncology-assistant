from src.graph.state import RAGState


def apply_output_guardrail(state: RAGState) -> dict:
    """Basic faithfulness check: ensure answer references at least one source."""
    answer = state.get("answer_professional", "")
    chunks = state.get("retrieved_chunks", [])

    if not chunks and answer:
        return {
            "output_blocked": True,
            "answer_professional": "Die Anfrage konnte nicht mit den verfügbaren Leitlinienabschnitten beantwortet werden.",
            "answer_plain": "Es wurden keine relevanten Informationen in den Leitlinien gefunden.",
        }
    return {"output_blocked": False}
