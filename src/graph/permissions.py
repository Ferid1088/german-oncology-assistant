"""Role-based access control (RBAC) for agent tools and external data sources.

Defines which tools each user role may invoke and which external sources
(web, pubmed) are permitted.  All three roles currently share identical tool
sets; the structure is kept so per-role restrictions can be added later without
changing the calling code.
"""

from __future__ import annotations


# All roles currently grant access to the same tool set.
# Separate the definitions so future role restrictions can be added per-role
# without touching is_tool_allowed().
ROLE_ALLOWED_TOOLS: dict[str, set[str]] = {
    "user": {
        "search_guidelines",
        "lookup_empfehlung",
        "compare_guidelines",
        "drug_class_lookup",
        "calculate_bmi",
        "pubmed_search",
        "web_search_snippets",
    },
    "professional": {
        "search_guidelines",
        "lookup_empfehlung",
        "compare_guidelines",
        "drug_class_lookup",
        "calculate_bmi",
        "pubmed_search",
        "web_search_snippets",
    },
    "admin": {
        "search_guidelines",
        "lookup_empfehlung",
        "compare_guidelines",
        "drug_class_lookup",
        "calculate_bmi",
        "pubmed_search",
        "web_search_snippets",
    },
}


def is_source_allowed(state: dict, source: str) -> bool:
    """Return True if *source* is permitted for this request.

    Args:
        state: Current RAGState dict containing ``allowed_sources``.
        source: Source identifier to check, e.g. ``"web"`` or ``"pubmed"``.

    Returns:
        True when ``allowed_sources`` is empty (no restriction) or contains
        *source*.  False otherwise.
    """
    allowed_sources = set(state.get("allowed_sources", []))
    return source in allowed_sources if allowed_sources else True


def is_tool_allowed(state: dict, tool_name: str) -> bool:
    """Return True if the user's role permits calling *tool_name*.

    Falls back to the ``"user"`` role policy when the state's ``user_role``
    value is not found in ``ROLE_ALLOWED_TOOLS``.

    Args:
        state: Current RAGState dict containing ``user_role``.
        tool_name: Name of the tool function to check.

    Returns:
        True when the role's allowed set contains *tool_name*.
    """
    role = state.get("user_role", "user")
    return tool_name in ROLE_ALLOWED_TOOLS.get(role, ROLE_ALLOWED_TOOLS["user"])
