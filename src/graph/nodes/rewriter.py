import json
import os
from openai import OpenAI
from src.graph.state import RAGState
from src.graph.messages import get_message_content, get_message_role
from src.prompts.rewriter import build_ambiguity_prompt_messages

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")

_VALID_INTENTS = {"factual", "recommendation", "comparison", "external"}
_VALID_GUIDELINES = {"mamma", "krk", "lunge", "prosta", ""}
_VALID_GRADES = {"A", "B", "0", ""}
_VALID_CHUNK_TYPES = {"recommendation", "section", ""}
_VALID_MISSING_DIMENSIONS = {
    "tumor_entity",
    "histology",
    "disease_stage",
    "therapy_setting",
    "line_of_therapy",
    "molecular_subtype",
    "biomarker_status",
    "risk_group",
    "patient_subgroup",
    "treatment_goal",
}


def _decompose_query(query: str, intent: str) -> list[str]:
    pieces: list[str] = []
    lowered = query.lower()
    if intent == "comparison" and " und " in lowered:
        pieces = [p.strip() for p in query.split(" und ") if p.strip()]
    elif any(marker in lowered for marker in ["sowie", "außerdem", "zusätzlich"]):
        import re

        pieces = [p.strip() for p in re.split(r"\b(?:sowie|außerdem|zusätzlich)\b", query, flags=re.IGNORECASE) if p.strip()]

    deduped: list[str] = []
    seen = set()
    for item in pieces:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```", 1)[1].lstrip("json").strip()
        if "```" in text:
            text = text.split("```", 1)[0].strip()
    return text


def _default_result(state: RAGState) -> dict:
    return {
        "rewritten_query": state["user_query"],
        "metadata_filters": dict(state.get("metadata_filters", {})),
        "intent": "factual",
        "query_decomposition": [],
        "requires_clarification": False,
        "missing_clinical_dimensions": [],
        "clarification_rationale": None,
        "expected_clarification": None,
    }


def _contains_clarification_marker(text: str) -> bool:
    content = (text or "").strip().lower()
    if not content:
        return False

    clarification_markers = [
        "ich brauche vor der leitlinienrecherche noch eine präzisierung",
        "bitte präzisieren sie",
        "präzisieren sie ihre klinische frage",
    ]

    return any(marker in content for marker in clarification_markers)


def _conversation_already_contains_clarification(messages: list, prior_answers: list[str] | None = None) -> bool:
    prior_answers = prior_answers or []

    for answer in prior_answers:
        if _contains_clarification_marker(answer):
            return True

    if not messages:
        return False

    for message in messages:
        if get_message_role(message) != "assistant":
            continue
        if _contains_clarification_marker(get_message_content(message) or ""):
            return True
    return False


def rewrite_query(state: RAGState, client: OpenAI | None = None) -> dict:
    """Single LLM call: rewrite query + extract filters + detect ambiguity."""
    c = client or _client()

    history = state.get("messages", [])
    history_block = ""
    if history:
        lines = "\n".join(
            f"{get_message_role(message)}: {get_message_content(message)}"
            for message in history[-4:]
            if get_message_content(message)
        )
        history_block = f"Gesprächsverlauf:\n{lines}\n\n"

    try:
        prompt_messages = build_ambiguity_prompt_messages(
            history_block=history_block,
            query=state["user_query"],
        )
        resp = c.chat.completions.create(
            model=CHEAP_MODEL,
            messages=prompt_messages,
            max_tokens=400,
        )
        raw = _strip_code_fence(resp.choices[0].message.content or "")
        parsed = json.loads(raw)
    except Exception:
        return _default_result(state)

    rewritten = str(parsed.get("rewritten_query") or state["user_query"]).strip() or state["user_query"]

    filters = {}
    gid = parsed.get("guideline_id", "")
    if gid in _VALID_GUIDELINES and gid:
        filters["guideline_id"] = gid
    grade = parsed.get("grade", "")
    if grade in _VALID_GRADES and grade:
        filters["grade"] = grade
    chunk_type = parsed.get("chunk_type", "")
    if chunk_type in _VALID_CHUNK_TYPES and chunk_type:
        filters["chunk_type"] = chunk_type

    intent = parsed.get("intent", "factual")
    if intent not in _VALID_INTENTS:
        intent = "factual"

    requires_clarification = bool(parsed.get("requires_clarification", False))
    raw_missing_dimensions = parsed.get("missing_clinical_dimensions") or []
    missing_dimensions = [
        value for value in raw_missing_dimensions
        if isinstance(value, str) and value in _VALID_MISSING_DIMENSIONS
    ]
    clarification_rationale = parsed.get("clarification_rationale")
    if not isinstance(clarification_rationale, str) or not clarification_rationale.strip():
        clarification_rationale = None
    else:
        clarification_rationale = clarification_rationale.strip()

    expected_clarification = parsed.get("expected_clarification")
    if not isinstance(expected_clarification, str) or not expected_clarification.strip():
        expected_clarification = None
    else:
        expected_clarification = expected_clarification.strip()

    prior_answers = [
        state.get("prior_answer_professional", ""),
        state.get("prior_answer_plain", ""),
    ]
    already_asked_for_clarification = _conversation_already_contains_clarification(history, prior_answers)

    if not requires_clarification:
        missing_dimensions = []
        clarification_rationale = None
        expected_clarification = None
    elif already_asked_for_clarification:
        requires_clarification = False
        missing_dimensions = []
        clarification_rationale = None
        expected_clarification = None
    elif expected_clarification is None:
        expected_clarification = "Bitte präzisieren Sie Ihre klinische Frage, damit ich gezielt in den Leitlinien suchen kann."

    # Merge with any filters the user already selected via the UI
    ui_filters = state.get("metadata_filters", {})
    merged_filters = {**filters, **ui_filters}

    return {
        "rewritten_query": rewritten,
        "metadata_filters": merged_filters,
        "intent": intent,
        "query_decomposition": _decompose_query(rewritten, intent),
        "requires_clarification": requires_clarification,
        "missing_clinical_dimensions": missing_dimensions,
        "clarification_rationale": clarification_rationale,
        "expected_clarification": expected_clarification,
    }
