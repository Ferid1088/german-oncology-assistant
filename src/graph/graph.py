from langgraph.graph import StateGraph, END
from src.graph.state import RAGState
from src.graph.nodes.guardrail_input import apply_input_guardrail
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.turn_router import route_turn
from src.graph.nodes.agent import run_agent
from src.graph.nodes.confidence import check_confidence, needs_escalation
from src.graph.nodes.answer import generate_answer
from src.graph.nodes.guardrail_output import apply_output_guardrail
from src.retrieval.postprocess import top_unique_result_dicts


def _multi_query_escalation(state: RAGState) -> dict:
    """Lightweight multi-query/decomposition fallback within current architecture."""
    from src.tools.search_guidelines import search_guidelines_tool

    base_query = state.get("rewritten_query") or state.get("redacted_query") or state["user_query"]
    subqueries = state.get("query_decomposition") or []

    candidate_queries = [base_query]
    if subqueries:
        candidate_queries.extend(subqueries)
    candidate_queries.append(f"Leitlinienempfehlungen zu {base_query}")
    candidate_queries.append(f"Empfehlung Evidenz Therapie {base_query}")

    seen: set[str] = set()
    merged: list[dict] = []
    for q in candidate_queries:
        q = q.strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        hits = search_guidelines_tool(
            query=q,
            guideline_id=state.get("metadata_filters", {}).get("guideline_id"),
            grade=state.get("metadata_filters", {}).get("grade"),
            top_k=5,
        )
        merged.extend(hits)

    return {"retrieved_chunks": top_unique_result_dicts(merged, top_k=10)}


def _blocked_response(state: RAGState) -> dict:
    return {
        "answer_professional": state.get("input_block_reason", "Anfrage blockiert."),
        "answer_plain": state.get("input_block_reason", "Anfrage blockiert."),
        "citations": [],
        "disclaimer": "",
    }


def _route_after_guardrail(state: RAGState) -> str:
    return "blocked" if state["input_blocked"] else "rewrite"


def _route_after_rewrite(state: RAGState) -> str:
    return "turn_router"


def build_graph(checkpointer=None):
    builder = StateGraph(RAGState)

    builder.add_node("guardrail_input", apply_input_guardrail)
    builder.add_node("blocked", _blocked_response)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("turn_router", route_turn)
    builder.add_node("agent", run_agent)
    builder.add_node("confidence", check_confidence)
    builder.add_node("escalate", _multi_query_escalation)
    builder.add_node("answer", generate_answer)
    builder.add_node("guardrail_output", apply_output_guardrail)

    builder.set_entry_point("guardrail_input")
    builder.add_conditional_edges("guardrail_input", _route_after_guardrail, {
        "blocked": "blocked",
        "rewrite": "rewrite",
    })
    builder.add_edge("blocked", END)
    builder.add_conditional_edges("rewrite", _route_after_rewrite, {
        "turn_router": "turn_router",
    })
    builder.add_edge("turn_router", "agent")
    builder.add_edge("agent", "confidence")
    builder.add_conditional_edges("confidence", needs_escalation, {
        "escalate": "escalate",
        "answer": "answer",
    })
    builder.add_edge("escalate", "answer")
    builder.add_edge("answer", "guardrail_output")
    builder.add_edge("guardrail_output", END)

    return builder.compile(checkpointer=checkpointer)
