import os
import re
from openai import OpenAI
from src.graph.state import RAGState
from src.prompts.answer import EXTRACT_PROMPT, SYNTHESIZE_PROMPT

GEN_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-4o")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")

DISCLAIMER = (
    "\n\n---\n*Haftungsausschluss: Diese Informationen stammen aus den deutschen S3-Leitlinien "
    "und dienen ausschließlich zu Bildungszwecken. Sie ersetzen keine individuelle medizinische Beratung.*"
)
_NOT_FOUND = (
    "Die gestellte Frage kann mit den verfügbaren Leitlinienabschnitten nicht beantwortet werden. "
    "Bitte wenden Sie sich an einen Facharzt oder stellen Sie eine Frage zu den S3-Leitlinien für "
    "Mammakarzinom, Kolorektales Karzinom, Lungenkarzinom oder Prostatakarzinom."
)


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _extract(llm: OpenAI, query: str, context: str) -> str:
    """Stage 1: extract verbatim sentences from chunks that answer the query."""
    resp = llm.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(query=query, context=context)}],
        max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()


def _synthesize(llm: OpenAI, query: str, extracted: str, valid_numbers: str) -> str:
    """Stage 2: synthesize coherent answer from extracted sentences only."""
    resp = llm.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": SYNTHESIZE_PROMPT.format(
            query=query,
            extracted=extracted,
            valid_numbers=valid_numbers,
        )}],
        max_tokens=1500,
    )
    return resp.choices[0].message.content.strip()


def generate_answer(state: RAGState, client: OpenAI | None = None) -> dict:
    llm = client or _client()
    chunks = state.get("retrieved_chunks", [])
    if not chunks and state.get("followup_routing") == "memory":
        chunks = state.get("prior_retrieved_chunks", [])

    all_citations = [
        {
            "label": ch.get("citation_label") or f"[{i + 1}]",
            "chunk_id": ch.get("chunk_id", ""),
            "citation": ch["citation"],
            "source_filename": ch.get("source_filename", ""),
            "section_path": ch.get("section_path", []),
            "page_start": ch.get("page_start"),
            "page_end": ch.get("page_end"),
            "section_title": ch.get("section_title", ""),
            "guideline_id": ch.get("guideline_id", ""),
            "recommendation_id": ch.get("recommendation_id", ""),
            "recommendation_grade": ch.get("recommendation_grade", ""),
            "evidence_level": ch.get("evidence_level", ""),
            "reference_ids": ch.get("reference_ids", []),
            "contextual_header": ch.get("contextual_header", ""),
            "parent_chunk_id": ch.get("parent_chunk_id", ""),
            "is_opinion": False,
        }
        for i, ch in enumerate(chunks[:5])
    ]
    n = len(chunks[:5])
    context = "\n\n".join(
        f"[{i + 1}] {ch['citation']}: {ch['text'][:800]}"
        for i, ch in enumerate(chunks[:5])
    )

    if state.get("followup_routing") == "memory":
        prior_answer = state.get("prior_answer_professional", "")
        prior_plain = state.get("prior_answer_plain", "")
        if prior_answer or prior_plain:
            conversation_context = "\n\n".join(filter(None, [
                f"Vorherige fachliche Antwort: {prior_answer}",
                f"Vorherige einfache Antwort: {prior_plain}",
                f"Vorherige Turn-Intents: {', '.join(state.get('turn_intents', []))}",
            ]))
            context = "\n\n".join(filter(None, [conversation_context, context]))

    # Stage 1: extract verbatim sentences from chunks
    extracted = _extract(llm, state["user_query"], context)

    # If extraction found nothing relevant, refuse to generate
    if not extracted or extracted.strip().upper() == "NICHTS" or len(extracted.strip()) < 20:
        return {
            "answer_professional": _NOT_FOUND,
            "answer_plain": _NOT_FOUND,
            "citations": [],
            "disclaimer": DISCLAIMER,
        }

    # Stage 2: synthesize from extracted sentences only (no original chunks visible)
    valid_numbers = ", ".join(str(i) for i in range(1, n + 1))
    full = _synthesize(llm, state["user_query"], extracted, valid_numbers)

    if "**In einfachen Worten:**" in full:
        parts = full.split("**In einfachen Worten:**", 1)
        pro = parts[0].replace("**Fachliche Antwort:**", "").strip()
        plain = parts[1].strip()
    else:
        pro = full
        plain = ""

    # Collect every cited number across both parts
    used_indices: set[int] = set()
    for bracket_content in re.findall(r'\[([^\]]+)\]', full):
        for num in re.findall(r'\d+', bracket_content):
            used_indices.add(int(num))
    citations = [c for c in all_citations if int(c["label"][1:-1]) in used_indices]

    return {
        "answer_professional": pro,
        "answer_plain": plain,
        "citations": citations,
        "disclaimer": DISCLAIMER,
    }
