import re

from langgraph.graph import StateGraph, END
from src.graph.state import RAGState
from src.graph.nodes.guardrail_input import apply_input_guardrail
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.turn_router import route_turn
from src.graph.nodes.agent import run_agent
from src.graph.nodes.confidence import check_confidence, needs_escalation
from src.graph.nodes.answer import DISCLAIMER, generate_answer
from src.graph.nodes.guardrail_output import apply_output_guardrail
from src.graph.nodes.external_search import run_external_search
from src.retrieval.postprocess import top_unique_result_dicts
from src.telemetry import append_rag_step


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

    deduped = top_unique_result_dicts(merged, top_k=10)
    return {
        "retrieved_chunks": deduped,
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="escalate",
            status="ok" if deduped else "empty",
            summary=f"Escalation ran additional retrieval queries and produced {len(deduped)} chunk(s).",
            details={"candidate_queries": len(candidate_queries)},
        ),
    }


def _blocked_response(state: RAGState) -> dict:
    return {
        "answer_professional": state.get("input_block_reason", "Anfrage blockiert."),
        "answer_plain": state.get("input_block_reason", "Anfrage blockiert."),
        "citations": [],
        "disclaimer": "",
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="blocked",
            status="blocked",
            summary="The conversation ended because input guardrails blocked the request.",
        ),
    }


def _clarification_response(state: RAGState) -> dict:
    rationale = (state.get("clarification_rationale") or "").strip()
    followup = (state.get("expected_clarification") or "Bitte präzisieren Sie Ihre Anfrage.").strip()

    intro = "Ich brauche vor der Leitlinienrecherche noch eine Präzisierung Ihrer Frage."
    clarification_text = "\n\n".join(part for part in [intro, rationale, followup] if part)
    return {
        "requires_clarification": True,
        "missing_clinical_dimensions": list(state.get("missing_clinical_dimensions", [])),
        "clarification_rationale": rationale or None,
        "expected_clarification": followup,
        "answer_professional": clarification_text,
        "answer_plain": "",
        "citations": [],
        "disclaimer": "",
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="clarification",
            status="ok",
            summary="The system asked the user for more clinical detail before retrieval.",
            details={
                "expected_clarification": followup,
                "missing_clinical_dimensions": list(state.get("missing_clinical_dimensions", [])),
            },
        ),
    }


def _route_after_output(state: RAGState) -> str:
    if state.get("input_blocked") or state.get("output_blocked") or state.get("requires_clarification"):
        return "end"
    return "external_search"


def _route_after_guardrail(state: RAGState) -> str:
    return "blocked" if state["input_blocked"] else "rewrite"


def _route_after_rewrite(state: RAGState) -> str:
    if state.get("requires_clarification"):
        return "clarification"
    if _is_duplicate_of_prior_query(state):
        return "repeat_answer"
    return "turn_router"


def _normalize_query(value: str) -> str:
    text = (value or "").strip().casefold()
    text = re.sub(r"[^\w\säöüß]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_duplicate_of_prior_query(state: RAGState) -> bool:
    current = _normalize_query(state.get("rewritten_query") or state.get("redacted_query") or state.get("user_query", ""))
    previous = _normalize_query(state.get("prior_rewritten_query", ""))
    has_prior_answer = bool(state.get("prior_answer_professional") or state.get("prior_answer_plain"))
    return bool(current and previous and has_prior_answer and current == previous)


def _prior_disclaimer(state: RAGState) -> str:
    prior_rag_trace = state.get("prior_rag_trace", [])
    if isinstance(prior_rag_trace, list) and any(
        isinstance(step, dict) and step.get("name") == "answer" for step in prior_rag_trace
    ):
        return DISCLAIMER
    return ""


def _repeat_previous_answer_response(state: RAGState) -> dict:
    rewritten_query = state.get("rewritten_query") or state.get("redacted_query") or state.get("user_query", "")
    return {
        "answer_professional": state.get("prior_answer_professional", ""),
        "answer_plain": state.get("prior_answer_plain", ""),
        "citations": list(state.get("prior_citations", [])),
        "retrieved_chunks": list(state.get("prior_retrieved_chunks", [])),
        "external_search_snippets": list(state.get("prior_external_search_snippets", [])),
        "tool_calls_log": [],
        "disclaimer": _prior_disclaimer(state),
        "followup_routing": "memory",
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="repeat_answer",
            status="ok",
            summary="The previous answer was replayed because the rewritten query matched the previous intent.",
            details={
                "rewritten_query": rewritten_query,
                "prior_rewritten_query": state.get("prior_rewritten_query", ""),
            },
        ),
    }


def build_graph(checkpointer=None):
    builder = StateGraph(RAGState)

    builder.add_node("guardrail_input", apply_input_guardrail)
    builder.add_node("blocked", _blocked_response)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("clarification", _clarification_response)
    builder.add_node("repeat_answer", _repeat_previous_answer_response)
    builder.add_node("turn_router", route_turn)
    builder.add_node("agent", run_agent)
    builder.add_node("confidence", check_confidence)
    builder.add_node("escalate", _multi_query_escalation)
    builder.add_node("answer", generate_answer)
    builder.add_node("guardrail_output", apply_output_guardrail)
    builder.add_node("external_search", run_external_search)

    builder.set_entry_point("guardrail_input")
    builder.add_conditional_edges("guardrail_input", _route_after_guardrail, {
        "blocked": "blocked",
        "rewrite": "rewrite",
    })
    builder.add_edge("blocked", END)
    builder.add_conditional_edges("rewrite", _route_after_rewrite, {
        "clarification": "clarification",
        "repeat_answer": "repeat_answer",
        "turn_router": "turn_router",
    })
    builder.add_edge("clarification", END)
    builder.add_edge("repeat_answer", END)
    builder.add_edge("turn_router", "agent")
    builder.add_edge("agent", "confidence")
    builder.add_conditional_edges("confidence", needs_escalation, {
        "escalate": "escalate",
        "answer": "answer",
    })
    builder.add_edge("escalate", "answer")
    builder.add_edge("answer", "guardrail_output")
    builder.add_conditional_edges("guardrail_output", _route_after_output, {
        "external_search": "external_search",
        "end": END,
    })
    builder.add_edge("external_search", END)

    return builder.compile(checkpointer=checkpointer)
