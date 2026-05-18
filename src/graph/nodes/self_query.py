import json
import os
from openai import OpenAI
from src.graph.state import RAGState

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def extract_metadata_filters(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    prompt = (
        "Extrahiere Metadaten-Filter aus dieser Anfrage für eine Leitlinien-Datenbank. "
        "Antworte ausschließlich mit JSON:\n"
        '{"guideline_id": "" | "mamma" | "krk" | "lunge" | "prosta", "grade": "" | "A" | "B" | "0", "chunk_type": "" | "recommendation" | "section"}\n\n'
        f"Anfrage: {state['rewritten_query'] or state['user_query']}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    try:
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json")
        filters = json.loads(raw)
        return {"metadata_filters": {k: v for k, v in filters.items() if v}}
    except Exception:
        return {"metadata_filters": {}}
