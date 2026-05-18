from langgraph.graph import StateGraph, END
from src.graph.state import RAGState
from src.graph.nodes.guardrail_input import apply_input_guardrail
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.self_query import extract_metadata_filters
from src.graph.nodes.router import route_intent
from src.graph.nodes.agent import run_agent
from src.graph.nodes.confidence import check_confidence, needs_escalation
from src.graph.nodes.answer import generate_answer
from src.graph.nodes.guardrail_output import apply_output_guardrail


def _multi_query_escalation(state: RAGState) -> dict:
    """Simple multi-query fallback: run search with a broader reformulation."""
    from src.tools.search_guidelines import search_guidelines_tool
    query = state.get("rewritten_query") or state["user_query"]
    broader_query = f"Leitlinienempfehlungen zu {query}"
    chunks = search_guidelines_tool(query=broader_query, top_k=5)
    return {"retrieved_chunks": chunks}


def _blocked_response(state: RAGState) -> dict:
    return {
        "answer_professional": state.get("input_block_reason", "Anfrage blockiert."),
        "answer_plain": state.get("input_block_reason", "Anfrage blockiert."),
        "citations": [],
        "disclaimer": "",
    }


def _route_after_guardrail(state: RAGState) -> str:
    return "blocked" if state["input_blocked"] else "rewrite"


def build_graph(checkpointer=None):
    builder = StateGraph(RAGState)

    builder.add_node("guardrail_input", apply_input_guardrail)
    builder.add_node("blocked", _blocked_response)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("self_query", extract_metadata_filters)
    builder.add_node("router", route_intent)
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
    builder.add_edge("rewrite", "self_query")
    builder.add_edge("self_query", "router")
    builder.add_edge("router", "agent")
    builder.add_edge("agent", "confidence")
    builder.add_conditional_edges("confidence", needs_escalation, {
        "escalate": "escalate",
        "answer": "answer",
    })
    builder.add_edge("escalate", "answer")
    builder.add_edge("answer", "guardrail_output")
    builder.add_edge("guardrail_output", END)

    return builder.compile(checkpointer=checkpointer)
