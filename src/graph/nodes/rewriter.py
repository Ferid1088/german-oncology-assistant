import json
import os
from openai import OpenAI
from src.graph.state import RAGState
from src.prompts.rewriter import ANALYZE_QUERY

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")

_VALID_INTENTS = {"factual", "recommendation", "comparison", "external"}
_VALID_GUIDELINES = {"mamma", "krk", "lunge", "prosta", ""}
_VALID_GRADES = {"A", "B", "0", ""}
_VALID_CHUNK_TYPES = {"recommendation", "section", ""}


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def rewrite_query(state: RAGState, client: OpenAI | None = None) -> dict:
    """Single LLM call: rewrite query + extract metadata filters + classify intent."""
    c = client or _client()

    history = state.get("messages", [])
    history_block = ""
    if history:
        lines = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        history_block = f"Gesprächsverlauf:\n{lines}\n\n"

    prompt = ANALYZE_QUERY.format(history_block=history_block, query=state["user_query"])

    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
        if "```" in raw:
            raw = raw[:raw.index("```")]

    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "rewritten_query": state["user_query"],
            "metadata_filters": {},
            "intent": "factual",
        }

    rewritten = parsed.get("rewritten_query") or state["user_query"]

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

    # Merge with any filters the user already selected via the UI
    ui_filters = state.get("metadata_filters", {})
    merged_filters = {**filters, **ui_filters}

    return {
        "rewritten_query": rewritten,
        "metadata_filters": merged_filters,
        "intent": intent,
    }
