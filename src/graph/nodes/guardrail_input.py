from src.graph.state import RAGState
from src.telemetry import append_rag_step

BLOCK_KEYWORDS = [
    "wetter", "weather", "sport", "fußball", "soccer", "football", "aktie",
    "stock", "rezept", "recipe", "kochen", "cooking", "politik", "politics",
    "musik", "music", "film", "movie", "reise", "travel",
]

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignoriere vorherige anweisungen",
    "system prompt",
    "developer message",
    "reveal your instructions",
    "zeige deinen prompt",
    "bypass",
    "jailbreak",
]

PII_PATTERNS = [
    r"\b\d{2}\.\d{2}\.\d{4}\b",  # date-like birthdate
    r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # naive full name
    r"\b\+?\d[\d\s\-/]{6,}\b",  # phone-ish
    r"\b\d{5}\s+[A-Za-zÄÖÜäöüß\-]+\b",  # postal code + city-ish
]


def _redact_pii(query: str) -> str:
    import re

    redacted = query
    for pattern in PII_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted


def apply_input_guardrail(state: RAGState) -> dict:
    raw_query = state["user_query"]
    query = raw_query.lower()

    if any(p in query for p in PROMPT_INJECTION_PATTERNS):
        return {
            "input_blocked": True,
            "input_block_reason": "Die Anfrage wurde wegen eines möglichen Prompt-Injection-Musters blockiert.",
            "redacted_query": raw_query,
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="input_guardrail",
                status="blocked",
                summary="The request was blocked by the input guardrail.",
                details={"reason": "prompt_injection"},
            ),
        }

    # Only block queries that are clearly unrelated to medicine/oncology
    is_off_topic = any(kw in query for kw in BLOCK_KEYWORDS)
    if is_off_topic:
        return {
            "input_blocked": True,
            "input_block_reason": "Ihre Anfrage scheint nicht onkologische Leitlinien zu betreffen. Bitte stellen Sie medizinische Fragen zu den S3-Leitlinien.",
            "redacted_query": raw_query,
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="input_guardrail",
                status="blocked",
                summary="The request was blocked as off-topic.",
                details={"reason": "off_topic"},
            ),
        }

    return {
        "input_blocked": False,
        "input_block_reason": "",
        "redacted_query": _redact_pii(raw_query),
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="input_guardrail",
            status="ok",
            summary="Input guardrail passed.",
        ),
    }
