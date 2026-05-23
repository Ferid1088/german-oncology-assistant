"""Turn router node: classifies user intent and decides whether to reuse memory or retrieve.

Runs after the rewriter and before the agent.  Uses a two-stage strategy:
1. **Heuristic fast-path** — regex patterns match common German follow-up phrases
   (e.g. "kürzer", "in 3 Sätzen") and return a routing decision without an LLM call.
2. **LLM fallback** — Gemini 2.5 Flash classifies the turn intent from the
   conversation history and returns a JSON payload with ``turn_intents`` and
   ``followup_routing``.  Falls back to ``retrieve`` on any parse error.

The ``followup_routing`` value controls the agent node:
- ``"memory"`` — skip full retrieval, reuse prior answer and chunks.
- ``"retrieve"`` — run the full tool-calling loop.
"""

import json
import os
import re
import time
from openai import OpenAI
from src.graph.state import RAGState
from src.graph.messages import get_message_content, get_message_role
from src.prompts.turn_router import TURN_ROUTER_PROMPT
from src.telemetry import append_rag_step, merge_token_usage, usage_from_response

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
# Allowed intent labels returned by the LLM; unknown values are discarded.
_VALID_INTENTS = {"clarify", "simplify", "expand", "refine", "new_query"}
# Allowed routing values; unknown values fall back to "retrieve" for safety.
_VALID_ROUTES = {"memory", "retrieve"}


def _heuristic_followup_route(query: str, has_history: bool) -> dict | None:
    """Return a routing decision if the query matches a known follow-up pattern.

    Avoids an LLM call for common German phrases that unambiguously signal a
    reformulation request (simplify, shorten, expand) on the prior answer.

    Args:
        query: Raw user query for the current turn.
        has_history: True when the conversation has at least one prior turn.

    Returns:
        A partial state dict with ``turn_intents`` and ``followup_routing``
        if a pattern matches, or ``None`` to fall through to the LLM.
    """
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
    """Return an OpenRouter-backed OpenAI client."""
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _history_block(messages: list) -> str:
    """Render the last 6 conversation messages as a plain-text block for the prompt.

    Args:
        messages: List of LangChain message objects from ``RAGState.messages``.

    Returns:
        Newline-joined ``role: content`` pairs, or an empty string if no messages.
    """
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
    """LangGraph node: classify turn intent and select memory vs. retrieve routing.

    Priority:
    1. No history → always ``retrieve``.
    2. Heuristic match → return immediately without LLM call.
    3. LLM classification → Gemini 2.5 Flash parses intent from conversation history.
    4. LLM error → safe fallback to ``retrieve``.

    The ``"new_query"`` intent always forces ``"retrieve"`` regardless of what
    the LLM returns for ``followup_routing``.

    Args:
        state: Current RAGState; reads ``messages``, ``user_query``, ``rag_trace``,
               ``token_usage``.
        client: Optional pre-built OpenAI client (used in tests).

    Returns:
        Partial RAGState update with ``turn_intents``, ``followup_routing``,
        ``token_usage``, and an updated ``rag_trace``.
    """
    messages = state.get("messages", [])
    if not messages:
        return {
            "turn_intents": ["new_query"],
            "followup_routing": "retrieve",
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="turn_router",
                status="ok",
                summary="No prior conversation history was available, so the query was treated as a new retrieval turn.",
            ),
        }

    heuristic = _heuristic_followup_route(state["user_query"], has_history=bool(messages))
    if heuristic:
        return {
            **heuristic,
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="turn_router",
                status="ok",
                summary=f"Turn router used a heuristic and selected '{heuristic['followup_routing']}'.",
                details={"turn_intents": heuristic.get("turn_intents", [])},
            ),
        }

    c = client or _client()
    prompt = TURN_ROUTER_PROMPT.format(
        history_block=_history_block(messages),
        query=state["user_query"],
    )

    try:
        started = time.perf_counter()
        resp = c.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
            if "```" in raw:
                raw = raw[:raw.index("```")]
        parsed = json.loads(raw)
    except Exception:
        return {
            "turn_intents": ["new_query"],
            "followup_routing": "retrieve",
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="turn_router",
                status="error",
                summary="Turn routing fell back to retrieval after an LLM parsing error.",
            ),
        }

    intents = [intent for intent in parsed.get("turn_intents", []) if intent in _VALID_INTENTS]
    if not intents:
        intents = ["new_query"]

    routing = parsed.get("followup_routing", "retrieve")
    if routing not in _VALID_ROUTES:
        routing = "retrieve"

    if "new_query" in intents:
        routing = "retrieve"

    usage = usage_from_response(resp, model=CHEAP_MODEL, step="turn_router", duration_ms=duration_ms)
    return {
        "turn_intents": intents,
        "followup_routing": routing,
        "token_usage": merge_token_usage(state.get("token_usage", {}), usage),
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="turn_router",
            status="ok",
            summary=f"Turn router selected '{routing}' routing.",
            details={"turn_intents": intents, "followup_routing": routing},
            duration_ms=duration_ms,
        ),
    }