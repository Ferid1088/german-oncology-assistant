"""Telemetry utilities: token usage tracking, cost calculation, and tool summarisation.

Token usage is aggregated across all LLM calls in a pipeline run and stored
in ``RAGState.token_usage``.  Cost is calculated from the pricing table at the
end of each call using the model-specific rates below.

``append_rag_step`` builds the ``rag_trace`` list that records every node's name,
status, summary, duration, and optional detail dict for debugging and UI display.

``summarize_tool_result`` converts raw tool output into a short human-readable
summary string and result-count for the tool call log shown in the UI.
"""

from __future__ import annotations

from typing import Any


# USD cost per 1M tokens for models available via OpenRouter.
# Update this table when adding new models or when pricing changes.
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "openai/gpt-4o": {"input": 5.0, "output": 15.0},
    "google/gemini-2.5-flash": {"input": 0.3, "output": 2.5},
}


def empty_token_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "currency": "USD",
        "calls": [],
    }


def merge_token_usage(*payloads: object) -> dict[str, Any]:
    merged = empty_token_usage()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        merged["input_tokens"] += int(payload.get("input_tokens", 0) or 0)
        merged["output_tokens"] += int(payload.get("output_tokens", 0) or 0)
        merged["total_tokens"] += int(payload.get("total_tokens", 0) or 0)
        merged["cost_usd"] = round(merged["cost_usd"] + float(payload.get("cost_usd", 0.0) or 0.0), 6)
        calls = payload.get("calls", [])
        if isinstance(calls, list):
            merged["calls"].extend(calls)
    return merged


def usage_from_response(response: object, *, model: str, step: str, duration_ms: float | None = None) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))

    pricing = MODEL_PRICING_USD_PER_1M.get(model, {})
    cost_usd = (
        (input_tokens / 1_000_000) * float(pricing.get("input", 0.0))
        + (output_tokens / 1_000_000) * float(pricing.get("output", 0.0))
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 6),
        "currency": "USD",
        "calls": [
            {
                "step": step,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost_usd, 6),
                "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
            }
        ],
    }


def append_rag_step(
    existing_steps: object,
    *,
    name: str,
    status: str,
    summary: str,
    details: dict[str, Any] | None = None,
    duration_ms: float | None = None,
) -> list[dict[str, Any]]:
    steps = list(existing_steps) if isinstance(existing_steps, list) else []
    step = {
        "name": name,
        "status": status,
        "summary": summary,
        "details": details or {},
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
    }
    steps.append(step)
    return steps


def _preview_lines(items: list[str], limit: int = 3) -> list[str]:
    return [item for item in items[:limit] if item]


def summarize_tool_result(name: str, result: object) -> tuple[str, list[str], int | None, str]:
    if isinstance(result, dict) and result.get("error"):
        return (str(result.get("error")), [], None, "error")

    if name == "search_guidelines" and isinstance(result, list):
        preview = _preview_lines([
            " · ".join(part for part in [str(item.get("guideline_id", "")).upper(), str(item.get("section_title", ""))] if part).strip()
            for item in result
        ])
        count = len(result)
        return (f"Guideline search returned {count} result(s).", preview, count, "ok")

    if name == "lookup_empfehlung" and isinstance(result, dict):
        rec_id = result.get("recommendation_id") or "unknown"
        title = result.get("section_title") or "Recommendation"
        return (f"Recommendation {rec_id} loaded.", _preview_lines([str(title)]), 1, "ok")

    if name == "compare_guidelines" and isinstance(result, dict):
        comparison = result.get("comparison") or result.get("summary") or "Comparison prepared."
        return ("Guideline comparison prepared.", _preview_lines([str(comparison)]), None, "ok")

    if name == "drug_class_lookup" and isinstance(result, dict):
        mentions = result.get("mentions") or result.get("results") or []
        count = len(mentions) if isinstance(mentions, list) else None
        preview = _preview_lines([
            str(item.get("section_title") or item.get("guideline_id") or "Match")
            for item in mentions[:3]
        ]) if isinstance(mentions, list) else []
        return (f"Drug-class lookup found {count or 0} mention(s).", preview, count, "ok")

    if name == "calculate_bmi" and isinstance(result, dict):
        bmi = result.get("bmi")
        category = result.get("category") or "BMI calculated"
        preview = [f"BMI {bmi}" if bmi is not None else str(category)]
        return (str(category), _preview_lines(preview), 1, "ok")

    if name == "pubmed_search" and isinstance(result, dict):
        entries = result.get("results", [])
        preview = _preview_lines([str(item.get("title", "")) for item in entries if isinstance(item, dict)])
        count = len(entries) if isinstance(entries, list) else 0
        return (f"PubMed search returned {count} paper(s).", preview, count, "ok")

    if name == "web_search_snippets" and isinstance(result, dict):
        entries = result.get("results", [])
        preview = _preview_lines([str(item.get("snippet", "")) for item in entries if isinstance(item, dict)])
        count = len(entries) if isinstance(entries, list) else 0
        return (f"Web search returned {count} snippet(s).", preview, count, "ok")

    if isinstance(result, list):
        return (f"{name} returned {len(result)} item(s).", _preview_lines([str(item) for item in result]), len(result), "ok")
    if isinstance(result, dict):
        preview = _preview_lines([f"{key}: {value}" for key, value in result.items() if key != "error"])
        return (f"{name} returned structured data.", preview, None, "ok")
    if result is None:
        return (f"{name} returned no data.", [], 0, "ok")
    return (f"{name} completed.", _preview_lines([str(result)]), None, "ok")