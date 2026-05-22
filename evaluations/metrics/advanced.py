"""Advanced structural and fidelity-oriented evaluation metrics."""

from __future__ import annotations

import re


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _combined_answer(resp: dict) -> str:
    return "\n\n".join(
        part for part in [resp.get("answer_professional", ""), resp.get("answer_plain", "")]
        if part
    ).strip()


def _contains_clarification_marker(text: str | None) -> bool:
    value = _normalize(text)
    if not value:
        return False
    markers = [
        "ich brauche vor der leitlinienrecherche noch eine präzisierung",
        "bitte präzisieren sie",
        "präzisieren sie ihre",
    ]
    return any(marker in value for marker in markers)


def _contains_citation_marker(text: str | None) -> bool:
    return bool(re.search(r"\[(?:\d+(?:\s*,\s*\d+)*)\]", text or ""))


def answer_presence(item: dict, resp: dict) -> bool | None:
    if item.get("expected_behavior") in {"refuse", "warn"}:
        return None
    return bool(_combined_answer(resp))


def citation_presence(item: dict, resp: dict) -> bool | None:
    should_expect_citations = (
        bool(item.get("required_citations"))
        or bool(item.get("gold_chunk_ids"))
        or (
            item.get("expected_behavior") == "answer"
            and not item.get("requires_clarification")
            and not item.get("should_refuse")
        )
    )
    if not should_expect_citations:
        return None

    return bool(resp.get("citations")) or _contains_citation_marker(resp.get("answer_professional", ""))


def plain_language_present(item: dict, resp: dict) -> bool | None:
    expected_sections = set(item.get("expected_answer_sections") or [])
    if "plain_language" not in expected_sections:
        return None
    return bool((resp.get("answer_plain") or "").strip())


def expected_sections_coverage(item: dict, resp: dict) -> float | None:
    expected_sections = item.get("expected_answer_sections") or []
    if not expected_sections:
        return None

    actual_sections: set[str] = set()
    answer_professional = resp.get("answer_professional", "")
    answer_plain = resp.get("answer_plain", "")

    if answer_professional:
        actual_sections.update({"answer", "recommendation", "evidence"})
    if answer_plain:
        actual_sections.add("plain_language")
    if resp.get("citations") or _contains_citation_marker(answer_professional):
        actual_sections.add("citations")
    if resp.get("blocked"):
        actual_sections.update({"refusal", "warning"})
    if resp.get("safety_warning"):
        actual_sections.add("warning")
    if resp.get("requires_clarification") or _contains_clarification_marker(answer_professional):
        actual_sections.add("clarification")

    hits = sum(1 for section in expected_sections if section in actual_sections)
    return round(hits / len(expected_sections), 4)


def recommendation_metadata_match(item: dict, resp: dict) -> float | None:
    expected = item.get("expected_recommendation_metadata") or {}
    relevant = {k: str(v).strip() for k, v in expected.items() if v is not None and str(v).strip()}
    if not relevant:
        return None

    answer_text = _combined_answer(resp).lower()
    citations = resp.get("citations", []) or []

    hits = 0
    for key, expected_value in relevant.items():
        expected_lower = expected_value.lower()
        in_answer = expected_lower in answer_text
        in_citations = any(
            str(citation.get(key, "")).strip().lower() == expected_lower
            for citation in citations
        )
        if in_answer or in_citations:
            hits += 1

    return round(hits / len(relevant), 4)


def claim_verdict_correct(item: dict, resp: dict) -> bool | None:
    verdict = item.get("claim_verdict")
    if not verdict:
        return None

    text = _combined_answer(resp).lower()
    if verdict == "supported":
        return bool(text) and not resp.get("blocked", False)

    negative_markers = [
        "nicht",
        "kein",
        "keine",
        "widerspricht",
        "nicht belegt",
        "nicht in der leitlinie",
        "findet sich nicht",
    ]
    return bool(resp.get("blocked")) or any(marker in text for marker in negative_markers)


def compute_all(item: dict, resp: dict) -> dict:
    return {
        "answer_presence": answer_presence(item, resp),
        "citation_presence": citation_presence(item, resp),
        "plain_language_present": plain_language_present(item, resp),
        "expected_sections_coverage": expected_sections_coverage(item, resp),
        "recommendation_metadata_match": recommendation_metadata_match(item, resp),
        "claim_verdict_correct": claim_verdict_correct(item, resp),
    }