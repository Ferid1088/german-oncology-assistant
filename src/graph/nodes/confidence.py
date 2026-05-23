"""Confidence scoring node: decides whether retrieval quality is sufficient.

Uses the CrossEncoder reranker scores stored on each retrieved chunk to compute
a lightweight proxy for answer quality.  If confidence is too low, the graph
routes to the ``escalate`` node which runs additional retrieval queries.
"""

from src.graph.state import RAGState
from src.telemetry import append_rag_step

# Minimum acceptable mean reranker score across the top-3 chunks.
# Below this threshold the escalation node runs additional retrieval queries.
CONFIDENCE_THRESHOLD = 0.5
# Minimum number of retrieved chunks required before confidence is meaningful.
# Fewer than this triggers escalation regardless of individual scores.
LOW_RESULT_THRESHOLD = 2


def check_confidence(state: RAGState) -> dict:
    """Lightweight confidence: mean reranker score of top-3 chunks."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "confidence": 0.0,
            "escalation_reason": "no_results",
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="confidence",
                status="empty",
                summary="No retrieved chunks were available for confidence scoring.",
            ),
        }
    top_scores = [c["score"] for c in chunks[:3] if "score" in c]
    confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
    reason = ""
    if len(chunks) < LOW_RESULT_THRESHOLD:
        reason = "low_result_count"
    elif confidence < CONFIDENCE_THRESHOLD:
        reason = "low_score"
    status = "ok" if not reason else "empty"
    summary = "Confidence check completed."
    if reason == "low_result_count":
        summary = "Confidence was limited because only a few retrieval results were available."
    elif reason == "low_score":
        summary = "Confidence was limited because reranker scores were low."
    return {
        "confidence": confidence,
        "escalation_reason": reason,
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="confidence",
            status=status,
            summary=summary,
            details={"confidence": round(confidence, 4), "reason": reason or None, "chunk_count": len(chunks)},
        ),
    }


def needs_escalation(state: RAGState) -> str:
    """Routing function: returns 'escalate' or 'answer'."""
    return "escalate" if state["confidence"] < CONFIDENCE_THRESHOLD else "answer"
