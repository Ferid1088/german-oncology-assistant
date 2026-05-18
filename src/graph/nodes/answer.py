import os
from openai import OpenAI
from src.graph.state import RAGState

GEN_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-4o")
DISCLAIMER = (
    "\n\n---\n*Haftungsausschluss: Diese Informationen stammen aus den deutschen S3-Leitlinien "
    "und dienen ausschließlich zu Bildungszwecken. Sie ersetzen keine individuelle medizinische Beratung.*"
)


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def generate_answer(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        f"[{i+1}] {ch['citation']}: {ch['text'][:600]}"
        for i, ch in enumerate(chunks[:5])
    )
    citations = [
        {"label": f"[{i+1}]", "citation": ch["citation"], "source_filename": ch.get("source_filename", "")}
        for i, ch in enumerate(chunks[:5])
    ]

    prompt = (
        f"Du bist ein medizinischer Leitlinien-Assistent. Beantworte die Frage basierend AUSSCHLIEßLICH auf den folgenden Leitlinienabschnitten.\n\n"
        f"Quellen:\n{context}\n\n"
        f"Frage: {state['user_query']}\n\n"
        "Antworte in ZWEI Teilen:\n\n"
        "**Fachliche Antwort:**\n"
        "Verwende formale medizinische Terminologie auf Deutsch. Nenne Empfehlungsgrade (A/B/0) und Evidenzlevel. "
        "Zitiere Quellen inline als [1], [2] etc.\n\n"
        "**In einfachen Worten:**\n"
        "Erkläre dasselbe in klarer, verständlicher Sprache für Nicht-Mediziner. Kein Fachjargon."
    )
    resp = c.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )
    full = resp.choices[0].message.content.strip()

    pro = ""
    plain = ""
    if "**In einfachen Worten:**" in full:
        parts = full.split("**In einfachen Worten:**", 1)
        pro = parts[0].replace("**Fachliche Antwort:**", "").strip()
        plain = parts[1].strip()
    else:
        pro = full

    return {
        "answer_professional": pro,
        "answer_plain": plain,
        "citations": citations,
        "disclaimer": DISCLAIMER,
    }
