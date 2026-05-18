import os
from openai import OpenAI
from src.graph.state import RAGState

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
VALID_INTENTS = {"factual", "recommendation", "comparison", "external"}


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def route_intent(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    prompt = (
        "Klassifiziere die Anfrage als einen der folgenden Typen und antworte nur mit dem Typ:\n"
        "factual | recommendation | comparison | external\n\n"
        f"Anfrage: {state['rewritten_query'] or state['user_query']}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    intent = resp.choices[0].message.content.strip().lower()
    if intent not in VALID_INTENTS:
        intent = "factual"
    return {"intent": intent}
