from __future__ import annotations

from src.graph.permissions import is_source_allowed
from src.graph.state import RAGState
from src.telemetry import append_rag_step, summarize_tool_result
from src.tools.web_search import web_search_snippets_tool


def run_external_search(state: RAGState) -> dict:
    if state.get("input_blocked") or state.get("output_blocked") or state.get("requires_clarification"):
        return {
            "external_search_snippets": [],
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="external_search",
                status="skipped",
                summary="External web snippets were skipped because the turn was blocked or needed clarification.",
            ),
        }

    if not is_source_allowed(state, "web"):
        return {
            "external_search_snippets": [],
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="external_search",
                status="skipped",
                summary="External web snippets are disabled for this request.",
            ),
        }

    query = state.get("rewritten_query") or state.get("redacted_query") or state.get("user_query", "")
    result = web_search_snippets_tool(query=query, max_results=5)
    snippets = result.get("results", []) if isinstance(result, dict) else []
    summary, preview, count, status = summarize_tool_result("web_search_snippets", result)
    tool_entry = {
        "tool": "web_search_snippets",
        "args": {"query": query, "max_results": 5},
        "summary": summary,
        "preview": preview,
        "status": status,
        "result_count": count,
        "auto": True,
        "provider": result.get("provider") if isinstance(result, dict) else None,
    }
    return {
        "external_search_snippets": snippets,
        "tool_calls_log": state.get("tool_calls_log", []) + [tool_entry],
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="external_search",
            status="ok" if snippets else "empty",
            summary=summary,
            details={
                "provider": result.get("provider") if isinstance(result, dict) else None,
                "snippet_count": len(snippets),
                "disclosure": result.get("disclosure") if isinstance(result, dict) else None,
            },
        ),
    }