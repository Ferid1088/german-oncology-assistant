"""Lightweight text similarity metrics — no LLM or Ragas required."""

from __future__ import annotations
import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def token_overlap(reference: str, candidate: str) -> float:
    """F1 over shared tokens between reference and candidate."""
    if not reference or not candidate:
        return 0.0
    ref_tokens = _tokens(reference)
    cand_tokens = _tokens(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    common = ref_tokens & cand_tokens
    if not common:
        return 0.0
    precision = len(common) / len(cand_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def char_overlap(reference: str, candidate: str) -> float:
    """Character-level overlap ratio (candidate coverage of reference)."""
    if not reference or not candidate:
        return 0.0
    ref_chars = set(reference.lower())
    cand_chars = set(candidate.lower())
    common = ref_chars & cand_chars
    return len(common) / len(ref_chars)


def compute_all(reference: str, candidate: str) -> dict:
    return {
        "token_overlap_f1": token_overlap(reference, candidate),
        "char_overlap": char_overlap(reference, candidate),
    }
