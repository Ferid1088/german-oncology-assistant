import os
import json
import re
import time
from openai import OpenAI
from src.graph.state import RAGState
from src.prompts.answer import EXTRACT_PROMPT, SYNTHESIZE_PROMPT, MEMORY_REWRITE_PROMPT
from src.telemetry import append_rag_step, merge_token_usage, usage_from_response

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


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```", 1)[1].lstrip("json").strip()
        if "```" in text:
            text = text.split("```", 1)[0].strip()
    return text


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _truncate_sentences(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:limit]).strip()


def _basic_memory_rewrite(query: str, prior_answer: str, prior_plain: str) -> dict:
    q = query.lower()
    source = prior_plain if any(token in q for token in ["einfach", "einfacher", "einfachen worten"]) and prior_plain else (prior_answer or prior_plain)
    if not source:
        return {"answer_professional": "", "answer_plain": ""}

    sentence_limit = None
    sentence_match = re.search(r"\b(1|2|3|4|5)\s+s[äa]tze?n?\b", q)
    if sentence_match:
        sentence_limit = int(sentence_match.group(1))
    elif re.search(r"\bein(?:em|e)?\s+satz\b", q):
        sentence_limit = 1
    elif "zwei sätze" in q:
        sentence_limit = 2

    if sentence_limit is not None:
        source = _truncate_sentences(source, sentence_limit)
    elif any(token in q for token in ["kürzer", "knapper", "zusammenfass"]):
        source = _truncate_sentences(source, 2)

    if any(token in q for token in ["einfach", "einfacher", "einfachen worten"]):
        return {"answer_professional": "", "answer_plain": source}
    return {"answer_professional": source, "answer_plain": ""}


def _extract(llm: OpenAI, query: str, context: str) -> tuple[str, dict]:
    """Stage 1: extract verbatim sentences from chunks that answer the query."""
    started = time.perf_counter()
    resp = llm.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(query=query, context=context)}],
        max_tokens=1200,
    )
    duration_ms = (time.perf_counter() - started) * 1000
    return resp.choices[0].message.content.strip(), usage_from_response(resp, model=CHEAP_MODEL, step="answer_extract", duration_ms=duration_ms)


def _synthesize(llm: OpenAI, query: str, extracted: str, valid_numbers: str) -> tuple[str, dict]:
    """Stage 2: synthesize coherent answer from extracted sentences only."""
    started = time.perf_counter()
    resp = llm.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": SYNTHESIZE_PROMPT.format(
            query=query,
            extracted=extracted,
            valid_numbers=valid_numbers,
        )}],
        max_tokens=1500,
    )
    duration_ms = (time.perf_counter() - started) * 1000
    return resp.choices[0].message.content.strip(), usage_from_response(resp, model=GEN_MODEL, step="answer_synthesize", duration_ms=duration_ms)


def _rewrite_from_memory(
    llm: OpenAI,
    query: str,
    prior_answer: str,
    prior_plain: str,
    turn_intents: list[str],
    valid_numbers: str,
) -> dict:
    try:
        started = time.perf_counter()
        resp = llm.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": MEMORY_REWRITE_PROMPT.format(
                query=query,
                previous_answer=prior_answer,
                previous_plain=prior_plain,
                turn_intents=", ".join(turn_intents) or "keine",
                valid_numbers=valid_numbers or "keine",
            )}],
            max_tokens=900,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        raw = _strip_code_fence(resp.choices[0].message.content or "")
        parsed = json.loads(raw)
        answer_professional = str(parsed.get("answer_professional", "") or "").strip()
        answer_plain = str(parsed.get("answer_plain", "") or "").strip()
        if answer_professional or answer_plain:
            return {
                "answer_professional": answer_professional,
                "answer_plain": answer_plain,
                "usage": usage_from_response(resp, model=GEN_MODEL, step="answer_memory_rewrite", duration_ms=duration_ms),
            }
    except Exception:
        pass

    fallback = _basic_memory_rewrite(query, prior_answer, prior_plain)
    fallback["usage"] = {}
    return fallback


def generate_answer(state: RAGState, client: OpenAI | None = None) -> dict:
    llm = client or _client()
    token_usage = state.get("token_usage", {})
    followup_memory = state.get("followup_routing") == "memory"
    chunks = state.get("retrieved_chunks", [])
    if not chunks and followup_memory:
        chunks = state.get("prior_retrieved_chunks", [])

    prior_citations = state.get("prior_citations", []) if followup_memory else []
    citation_source = prior_citations or chunks

    all_citations = [
        {
            "label": ch.get("citation_label") or f"[{i + 1}]",
            "chunk_id": ch.get("chunk_id", ""),
            "citation": ch["citation"],
            "source_filename": ch.get("source_filename", ""),
            "section_path": ch.get("section_path", []),
            "page_start": ch.get("page_start"),
            "page_end": ch.get("page_end"),
            "page_numbers": ch.get("page_numbers", []),
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
        for i, ch in enumerate(citation_source[:5])
    ]
    n = len(citation_source[:5])
    context = "\n\n".join(
        f"[{i + 1}] {ch['citation']}: {ch['text'][:800]}"
        for i, ch in enumerate(chunks[:5])
    )

    prior_answer = state.get("prior_answer_professional", "")
    prior_plain = state.get("prior_answer_plain", "")

    if followup_memory and (prior_answer or prior_plain):
        valid_numbers = ", ".join(
            sorted({c["label"][1:-1] for c in all_citations if c.get("label", "").startswith("[") and c.get("label", "").endswith("]")}, key=int)
        )
        rewritten = _rewrite_from_memory(
            llm,
            query=state["user_query"],
            prior_answer=prior_answer,
            prior_plain=prior_plain,
            turn_intents=state.get("turn_intents", []),
            valid_numbers=valid_numbers,
        )
        pro = rewritten.get("answer_professional", "")
        plain = rewritten.get("answer_plain", "")
        token_usage = merge_token_usage(token_usage, rewritten.get("usage", {}))
        full = "\n\n".join(filter(None, [pro, plain]))

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
            "token_usage": token_usage,
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="answer",
                status="ok",
                summary="Answer was rewritten from prior conversation memory.",
                details={"citation_count": len(citations), "followup_routing": "memory"},
            ),
        }

    if followup_memory and (prior_answer or prior_plain):
        conversation_context = "\n\n".join(filter(None, [
            f"Vorherige fachliche Antwort: {prior_answer}",
            f"Vorherige einfache Antwort: {prior_plain}",
            f"Vorherige Turn-Intents: {', '.join(state.get('turn_intents', []))}",
        ]))
        context = "\n\n".join(filter(None, [conversation_context, context]))

    # Stage 1: extract verbatim sentences from chunks
    extracted, extract_usage = _extract(llm, state["user_query"], context)
    token_usage = merge_token_usage(token_usage, extract_usage)

    # If extraction found nothing relevant, refuse to generate
    if not extracted or extracted.strip().upper() == "NICHTS" or len(extracted.strip()) < 20:
        return {
            "answer_professional": _NOT_FOUND,
            "answer_plain": _NOT_FOUND,
            "citations": [],
            "disclaimer": DISCLAIMER,
            "token_usage": token_usage,
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="answer",
                status="empty",
                summary="Answer generation stopped because extraction found no grounded evidence.",
            ),
        }

    # Stage 2: synthesize from extracted sentences only (no original chunks visible)
    valid_numbers = ", ".join(str(i) for i in range(1, n + 1))
    full, synth_usage = _synthesize(llm, state["user_query"], extracted, valid_numbers)
    token_usage = merge_token_usage(token_usage, synth_usage)

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
        "token_usage": token_usage,
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="answer",
            status="ok",
            summary="Answer generation completed from retrieved evidence.",
            details={"citation_count": len(citations), "followup_routing": state.get("followup_routing")},
        ),
    }
