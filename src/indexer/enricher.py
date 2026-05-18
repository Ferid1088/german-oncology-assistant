import json
import os
from openai import OpenAI

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
EMPTY_SEMANTIC = {"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": ""}


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def generate_contextual_header(
    chunk_text: str,
    section_path: list[str],
    guideline_title: str,
    client: OpenAI | None = None,
) -> str:
    c = client or _client()
    section_str = " > ".join(section_path)
    prompt = (
        f"Du analysierst einen Abschnitt der deutschen S3-Leitlinie '{guideline_title}', "
        f"Abschnitt {section_str}.\n\n"
        f"Erstelle einen kurzen Kontext-Header (1-2 Sätze), der erklärt, wo dieser Chunk "
        f"in der Leitlinie steht und was sein Hauptinhalt ist. Antworte nur mit dem Header, kein JSON.\n\n"
        f"Chunk:\n{chunk_text[:800]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def generate_hypothetical_questions(
    chunk_text: str,
    client: OpenAI | None = None,
) -> list[str]:
    c = client or _client()
    prompt = (
        "Generiere 2-3 medizinische Fragen auf Deutsch, die ein Arzt stellen würde, "
        "wenn er nach dem Inhalt dieses Leitlinien-Abschnitts sucht. "
        "Eine Frage pro Zeile, kein JSON.\n\n"
        f"Abschnitt:\n{chunk_text[:800]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    lines = resp.choices[0].message.content.strip().splitlines()
    return [l.strip("- •123.").strip() for l in lines if l.strip()]


def extract_semantic_metadata(
    chunk_text: str,
    client: OpenAI | None = None,
) -> dict:
    c = client or _client()
    prompt = (
        "Extrahiere semantische Metadaten aus diesem deutschen Leitlinien-Abschnitt. "
        "Antworte ausschließlich mit validem JSON (keine Erklärungen):\n"
        '{"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": ""}\n\n'
        f"Abschnitt:\n{chunk_text[:1000]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    try:
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return dict(EMPTY_SEMANTIC)
