import json
import os
import re
from openai import OpenAI
from src.graph.state import RAGState
from src.graph.messages import get_message_content, get_message_role
from src.prompts.turn_router import TURN_ROUTER_PROMPT

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
_VALID_INTENTS = {"clarify", "simplify", "expand", "refine", "new_query"}
_VALID_ROUTES = {"memory", "retrieve"}


def _heuristic_followup_route(query: str, has_history: bool) -> dict | None:
    if not has_history:
        return None

    q = query.strip().lower()
    if not q:
        return None

    if re.search(r"\b(nur|bitte)?\s*(in\s+)?(1|2|3|4|5|ein|eine|einem|zwei|drei|vier|fünf)\s+s[äa]tze?n?\b", q):
        return {"turn_intents": ["refine", "simplify"], "followup_routing": "memory"}

    if any(phrase in q for phrase in [
        "deine antwort zusammenfassen",
        "kannst du zusammenfassen",
        "bitte zusammenfassen",
        "zusammenfassen",
        "kürzer",
        "knapper",
        "vereinfache",
    ]):
        return {"turn_intents": ["simplify", "refine"], "followup_routing": "memory"}

    if any(phrase in q for phrase in [
        "in einfachen worten",
        "einfacher erklären",
        "einfach erklären",
        "einfacher erklären?",
    ]):
        return {"turn_intents": ["clarify", "simplify"], "followup_routing": "memory"}

    if any(phrase in q for phrase in [
        "mehr erklären",
        "genauer erklären",
        "ausführlicher",
        "kannst du mehr erklären",
        "kannst du das erklären",
        "kannst du das genauer erklären",
    ]):
        return {"turn_intents": ["clarify", "expand"], "followup_routing": "memory"}

    return None


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _history_block(messages: list) -> str:
    if not messages:
        return ""
    recent = messages[-6:]
    lines = []
    for message in recent:
        role = get_message_role(message)
        content = get_message_content(message)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def route_turn(state: RAGState, client: OpenAI | None = None) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"turn_intents": ["new_query"], "followup_routing": "retrieve"}

    heuristic = _heuristic_followup_route(state["user_query"], has_history=bool(messages))
    if heuristic:
        return heuristic

    c = client or _client()
    prompt = TURN_ROUTER_PROMPT.format(
        history_block=_history_block(messages),
        query=state["user_query"],
    )

    try:
        resp = c.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
            if "```" in raw:
                raw = raw[:raw.index("```")]
        parsed = json.loads(raw)
    except Exception:
        return {"turn_intents": ["new_query"], "followup_routing": "retrieve"}

    intents = [intent for intent in parsed.get("turn_intents", []) if intent in _VALID_INTENTS]
    if not intents:
        intents = ["new_query"]

    routing = parsed.get("followup_routing", "retrieve")
    if routing not in _VALID_ROUTES:
        routing = "retrieve"

    if "new_query" in intents:
        routing = "retrieve"

    return {
        "turn_intents": intents,
        "followup_routing": routing,
    }