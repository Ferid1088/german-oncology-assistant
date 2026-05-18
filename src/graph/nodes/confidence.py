from src.graph.state import RAGState

CONFIDENCE_THRESHOLD = 0.5


def check_confidence(state: RAGState) -> dict:
    """Lightweight confidence: mean reranker score of top-3 chunks."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"confidence": 0.0}
    top_scores = [c["score"] for c in chunks[:3] if "score" in c]
    confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
    return {"confidence": confidence}


def needs_escalation(state: RAGState) -> str:
    """Routing function: returns 'escalate' or 'answer'."""
    return "escalate" if state["confidence"] < CONFIDENCE_THRESHOLD else "answer"
