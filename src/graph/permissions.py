from __future__ import annotations


ROLE_ALLOWED_TOOLS: dict[str, set[str]] = {
    "user": {
        "search_guidelines",
        "lookup_empfehlung",
        "compare_guidelines",
        "drug_class_lookup",
        "calculate_bmi",
    },
    "professional": {
        "search_guidelines",
        "lookup_empfehlung",
        "compare_guidelines",
        "drug_class_lookup",
        "calculate_bmi",
        "pubmed_search",
    },
    "admin": {
        "search_guidelines",
        "lookup_empfehlung",
        "compare_guidelines",
        "drug_class_lookup",
        "calculate_bmi",
        "pubmed_search",
    },
}


def is_source_allowed(state: dict, source: str) -> bool:
    allowed_sources = set(state.get("allowed_sources", []))
    return source in allowed_sources if allowed_sources else True


def is_tool_allowed(state: dict, tool_name: str) -> bool:
    role = state.get("user_role", "user")
    return tool_name in ROLE_ALLOWED_TOOLS.get(role, ROLE_ALLOWED_TOOLS["user"])
