"""LLM-based enrichment functions for indexed guideline chunks.

Three enrichment steps run at index time (all via Gemini 2.5 Flash / OpenRouter):
1. ``generate_contextual_header`` — a 1-2 sentence summary of where in the guideline
   the chunk sits and what its main content is.  Prepended to the chunk text before
   embedding to improve semantic retrieval.
2. ``generate_hypothetical_questions`` — 2-3 clinical questions a doctor would ask to
   find this chunk.  Used for HyDE (Hypothetical Document Embeddings).
3. ``extract_semantic_metadata`` — structured JSON extraction of diseases, drugs,
   procedures, patient subgroups, and risk categories stored as dynamic Milvus fields.

All three functions fall back gracefully on LLM or JSON parse errors.
"""

import copy
import json
import os
from openai import OpenAI

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
EMPTY_SEMANTIC = {"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": []}


def _client() -> OpenAI:
    """Build an OpenAI-compatible client pointed at OpenRouter.

    Raises:
        ValueError: If ``OPENROUTER_API_KEY`` is not set in the environment.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def generate_contextual_header(
    chunk_text: str,
    section_path: list[str],
    guideline_title: str,
    client: OpenAI | None = None,
) -> str:
    """Generate a short contextual header for a chunk to improve embedding quality.

    The header is prepended to the chunk text before embedding so the vector
    representation reflects both the content and its location in the guideline.

    Args:
        chunk_text: Raw text content of the chunk (first 800 chars are sent to the LLM).
        section_path: Ordered list of section numbers forming the breadcrumb path.
        guideline_title: Full human-readable guideline title.
        client: Optional pre-built OpenAI client (injected in tests).

    Returns:
        A 1-2 sentence context header string, or an empty string on failure.
    """
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
    content = resp.choices[0].message.content
    return content.strip() if content else ""


def generate_hypothetical_questions(
    chunk_text: str,
    client: OpenAI | None = None,
) -> list[str]:
    """Generate 2-3 hypothetical clinical questions that this chunk would answer.

    Used for HyDE (Hypothetical Document Embeddings): the questions are embedded
    alongside the chunk to improve retrieval recall for clinical phrasings that differ
    from the formal guideline language.

    Args:
        chunk_text: Raw text content of the chunk (first 800 chars are sent to the LLM).
        client: Optional pre-built OpenAI client (injected in tests).

    Returns:
        A list of 2-3 question strings in German, or an empty list on failure.
    """
    c = client or _client()
    prompt = (
        "Generiere genau 2-3 medizinische Fragen auf Deutsch, die ein Arzt oder Kliniker "
        "stellen würde, wenn er gezielt nach dem Inhalt dieses Leitlinien-Abschnitts sucht. "
        "Regeln: Gib genau 2-3 Fragen aus. Eine Frage pro Zeile. "
        "Keine Nummerierung, keine Aufzählungszeichen, kein JSON. Nur die Fragen.\n\n"
        f"Abschnitt:\n{chunk_text[:800]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    content = resp.choices[0].message.content
    if not content:
        return []
    lines = content.strip().splitlines()
    return [l.strip("- •0123456789.").strip() for l in lines if l.strip()]


def extract_semantic_metadata(
    chunk_text: str,
    client: OpenAI | None = None,
) -> dict:
    """Extract structured semantic metadata from a chunk via LLM.

    Asks the model to return a JSON object with five lists: diseases, drugs,
    procedures, patient_subgroups, risk_category.  These are stored as dynamic
    Milvus fields to enable future metadata-based filtering.

    Falls back to ``EMPTY_SEMANTIC`` on any JSON parse or LLM error rather than
    raising, since enrichment failures should not abort the indexing pipeline.

    Args:
        chunk_text: Raw text content of the chunk (first 1000 chars are sent to the LLM).
        client: Optional pre-built OpenAI client (injected in tests).

    Returns:
        A dict with the five semantic lists, always with all five keys present.
    """
    c = client or _client()
    prompt = (
        "Extrahiere semantische Metadaten aus diesem deutschen Leitlinien-Abschnitt. "
        "Antworte ausschließlich mit validem JSON (keine Erklärungen):\n"
        '{"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": []}\n\n'
        f"Abschnitt:\n{chunk_text[:1000]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    try:
        raw = resp.choices[0].message.content
        if raw is None:
            return copy.deepcopy(EMPTY_SEMANTIC)
        raw = raw.strip()
        # Strip markdown code fences robustly
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw_parsed = json.loads(raw)
        if not isinstance(raw_parsed, dict):
            return copy.deepcopy(EMPTY_SEMANTIC)
        return {**copy.deepcopy(EMPTY_SEMANTIC), **raw_parsed}
    except (json.JSONDecodeError, IndexError, AttributeError):
        return copy.deepcopy(EMPTY_SEMANTIC)
