import os
from openai import OpenAI
from src.graph.state import RAGState

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def rewrite_query(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    history = state.get("messages", [])
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-4:]
    ) if history else ""

    prompt = (
        "Du bist ein Assistent für deutsche Onkologie-Leitlinien. "
        "Formuliere die folgende Anfrage als präzise medizinische Suchanfrage um. "
        "Berücksichtige den Gesprächsverlauf. Antworte nur mit der umformulierten Anfrage.\n\n"
        + (f"Verlauf:\n{history_text}\n\n" if history_text else "")
        + f"Anfrage: {state['user_query']}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return {"rewritten_query": resp.choices[0].message.content.strip()}
