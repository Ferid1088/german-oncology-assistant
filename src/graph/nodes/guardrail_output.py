"""Output safety guardrail: the last node before the response is returned.

Applies three sequential safety checks to the generated answer:
1. **Faithfulness check** — blocks any answer that has no retrieved evidence chunks,
   preventing hallucinated responses from reaching the user.
2. **Dosage safety** — blocks specific dosing figures (mg, mg/m², schema, cycles)
   unless at least one retrieved chunk directly contains that dosage information.
3. **Patient-specific warning** — does not block but adds a disclaimer when the
   query appears to be about a specific patient case rather than a general guideline
   question.

No LLM is used; all checks are regex-based for deterministic, low-latency safety.
"""

from __future__ import annotations

import re

from src.graph.state import RAGState
from src.telemetry import append_rag_step

# German regex patterns that indicate the query is about a specific patient,
# not a general guideline question.  Matches phrases like "mein Patient",
# "für diese Patientin", "individuell", etc.
_PATIENT_SPECIFIC_PATTERNS = [
    r"\bmein(?:e|em|en)?\s+patient(?:in)?\b",
    r"\bfür\s+diese[nr]?\s+patient(?:in)?\b",
    r"\bbei\s+diese[nr]?\s+patient(?:in)?\b",
    r"\bfür\s+meine[nr]?\s+patient(?:in)?\b",
    r"\bwas\s+soll\s+ich\s+(?:bei|für)\b",
    r"\bwelche\s+therapie\s+soll\b",
    r"\bindividuell(?:e|er|en)?\b",
]

_DOSAGE_QUERY_PATTERNS = [
    r"\bdosis\b",
    r"\bdosierung\b",
    r"\bmg\b",
    r"mg/m²",
    r"mg/m2",
    r"\bschema\b",
    r"\bregime\b",
    r"\bzyklen\b",
    r"\bq\d+w\b",
    r"alle\s+\d+\s+wochen",
]

_DOSAGE_EVIDENCE_PATTERN = re.compile(
    r"(\b\d+(?:[.,]\d+)?\s*(?:mg|g|µg|ml|mg/m²|mg/m2)\b|\bdosis\b|\bdosierung\b|\bschema\b|\bzyklen\b|\bq\d+w\b)",
    re.IGNORECASE,
)


def _matches_any(patterns: list[str], text: str) -> bool:
    """Return True if any regex pattern in *patterns* matches *text*.

    Args:
        patterns: List of regex pattern strings to test.
        text: Input string to search; safely handles None/empty.

    Returns:
        True on the first match; False if *text* is empty or no pattern matches.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def _is_patient_specific_query(query: str) -> bool:
    """Return True if the query appears to be about a specific patient case.

    Args:
        query: Raw user query string.
    """
    return _matches_any(_PATIENT_SPECIFIC_PATTERNS, query)


def _is_dosage_query(query: str) -> bool:
    """Return True if the query asks for concrete dosage or treatment regimen details.

    Args:
        query: Raw user query string.
    """
    return _matches_any(_DOSAGE_QUERY_PATTERNS, query)


def _has_direct_dosage_grounding(chunks: list[dict]) -> bool:
    """Return True if at least one retrieved chunk explicitly contains dosage figures.

    Searches chunk ``text``, ``citation``, and ``section_title`` fields for numeric
    dosage patterns (e.g. ``"200 mg"``, ``"q3w"``, ``"6 Zyklen"``).  Only allows
    dosage-related answers when direct evidence is present.

    Args:
        chunks: Retrieved chunk dicts from ``RAGState.retrieved_chunks``.
    """
    for chunk in chunks or []:
        haystacks = [
            str(chunk.get("text", "")),
            str(chunk.get("citation", "")),
            str(chunk.get("section_title", "")),
        ]
        if any(_DOSAGE_EVIDENCE_PATTERN.search(text) for text in haystacks if text):
            return True
    return False


def apply_output_guardrail(state: RAGState) -> dict:
    """LangGraph node: apply post-generation safety checks to the answer.

    Checks are evaluated in priority order:
    1. No chunks + non-empty answer → block (prevents hallucination).
    2. Dosage query without grounded dosage evidence → block with explanation.
    3. Patient-specific query → pass through but add a safety disclaimer.
    4. All checks pass → clear all safety fields.

    Args:
        state: Current RAGState; reads ``answer_professional``,
               ``retrieved_chunks``, ``user_query``, and ``rag_trace``.

    Returns:
        Partial RAGState update with ``output_blocked``, optional safety warning
        fields, and an updated ``rag_trace``.
    """
    answer = state.get("answer_professional", "")
    chunks = state.get("retrieved_chunks", [])
    query = state.get("user_query", "")

    patient_specific = _is_patient_specific_query(query)
    dosage_query = _is_dosage_query(query)
    has_dosage_grounding = _has_direct_dosage_grounding(chunks)

    if not chunks and answer:
        return {
            "output_blocked": True,
            "answer_professional": "Die Anfrage konnte nicht mit den verfügbaren Leitlinienabschnitten beantwortet werden.",
            "answer_plain": "Es wurden keine relevanten Informationen in den Leitlinien gefunden.",
            "safety_warning": None,
            "safety_explanation": None,
            "safety_title": None,
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="output_guardrail",
                status="blocked",
                summary="Output was blocked because no grounded evidence was available.",
            ),
        }

    if dosage_query and not has_dosage_grounding:
        return {
            "output_blocked": True,
            "answer_professional": (
                "Ich kann keine konkrete Dosierung oder ein exaktes Regime nennen, wenn diese Angaben nicht direkt "
                "durch die gefundenen Leitlinienstellen belegt sind. Ich kann stattdessen allgemeine Leitlinienprinzipien "
                "und die zitierten Therapieempfehlungen zusammenfassen."
            ),
            "answer_plain": (
                "Ich gebe keine genaue Dosis oder ein konkretes Behandlungsschema an, wenn das nicht direkt in den "
                "gefundenen Leitlinien steht."
            ),
            "citations": [],
            "safety_warning": "Dosage and regimen details were limited for safety.",
            "safety_explanation": (
                "Exact dosage or regimen advice is only allowed when it is directly grounded in the retrieved guideline evidence. "
                "This request was limited because the current evidence set does not clearly support a concrete dosing instruction."
            ),
            "safety_title": "Why was this limited?",
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="output_guardrail",
                status="blocked",
                summary="Output was limited because dosing details were not directly grounded.",
            ),
        }

    if patient_specific:
        return {
            "output_blocked": False,
            "safety_warning": "This answer is limited to general guideline information and is not an individual treatment decision.",
            "safety_explanation": (
                "The question appears patient-specific. The assistant can summarize general S3-guideline recommendations and cited evidence, "
                "but it should not present an individualized treatment decision for a specific patient."
            ),
            "safety_title": "Why was this limited?",
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="output_guardrail",
                status="ok",
                summary="Output guardrail added a patient-specific safety warning.",
            ),
        }

    return {
        "output_blocked": False,
        "safety_warning": None,
        "safety_explanation": None,
        "safety_title": None,
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="output_guardrail",
            status="ok",
            summary="Output guardrail passed.",
        ),
    }
