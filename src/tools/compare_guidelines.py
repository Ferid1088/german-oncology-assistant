"""Guideline comparison tool: runs parallel searches and returns side-by-side results.

Calls ``search_guidelines_tool`` twice with the same topic but different
``guideline_id`` values, then packages both result sets into a single payload.
Includes a ``summary_hint`` field that instructs GPT-4o to compare only on the
basis of the retrieved passages, not its training knowledge.
"""

import json  # noqa: F401 — kept for potential serialisation in callers

from src.tools.search_guidelines import search_guidelines_tool


def compare_guidelines_tool(
    topic: str,
    guideline_a: str,
    guideline_b: str,
    top_k: int = 3,
) -> dict:
    """Return a side-by-side comparison payload for two guidelines on the same topic."""
    results_a = search_guidelines_tool(query=topic, guideline_id=guideline_a, top_k=top_k)
    results_b = search_guidelines_tool(query=topic, guideline_id=guideline_b, top_k=top_k)

    return {
        "topic": topic,
        "guideline_a": guideline_a,
        "guideline_b": guideline_b,
        "results_a": results_a,
        "results_b": results_b,
        "summary_hint": (
            f"Vergleiche die Leitlinien {guideline_a} und {guideline_b} zum Thema '{topic}' "
            "nur auf Basis der gelieferten Fundstellen."
        ),
    }
