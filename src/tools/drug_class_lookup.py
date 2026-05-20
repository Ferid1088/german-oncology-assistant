from collections import defaultdict

from src.tools.search_guidelines import search_guidelines_tool


def drug_class_lookup_tool(
    substance_name: str,
    top_k_per_guideline: int = 5,
) -> dict:
    """Find mentions of a drug across guidelines and group them by guideline/grade."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for guideline_id in ["mamma", "krk", "lunge", "prosta"]:
        hits = search_guidelines_tool(
            query=substance_name,
            guideline_id=guideline_id,
            top_k=top_k_per_guideline,
        )
        for hit in hits:
            grouped[guideline_id].append(hit)

    return {
        "substance_name": substance_name,
        "matches_by_guideline": dict(grouped),
    }
