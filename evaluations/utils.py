"""Shared helpers for evaluation dataset readiness and item diagnostics."""

from __future__ import annotations

import json
from pathlib import Path


DATASET_FILENAMES = {
    "eval": "evaluation-dataset.json",
    "ab_test": "evaluation-dataset.json",
    "guardrail": "guardrail-dataset.json",
    "smoke": "smoke-dataset.json",
}


def load_json_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("items", []) if isinstance(payload, dict) else []


def default_dataset_path(root: Path, split: str = "eval") -> Path:
    return root / "data" / "eval" / DATASET_FILENAMES.get(split, DATASET_FILENAMES["eval"])


def resolve_dataset_path(root: Path, split: str, dataset: str | None = None) -> Path:
    if dataset:
        return Path(dataset)
    candidate = default_dataset_path(root, split)
    if candidate.exists():
        return candidate
    return default_dataset_path(root, "eval")


def discover_dataset_paths(dataset_path: Path) -> dict[str, Path]:
    base_dir = dataset_path.parent
    discovered: dict[str, Path] = {}
    for split, filename in DATASET_FILENAMES.items():
        candidate = base_dir / filename
        if candidate.exists():
            discovered[split] = candidate

    # A/B reuses the stable eval benchmark even when there is no dedicated file.
    if "eval" in discovered:
        discovered.setdefault("ab_test", discovered["eval"])
    return discovered


def normalize_scope_labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def bootstrap_reasons(item: dict) -> list[str]:
    reasons: list[str] = []

    source_chunk_ids = [str(value) for value in item.get("source_chunk_ids", [])]
    gold_chunk_ids = [str(value) for value in item.get("gold_chunk_ids", [])]
    citation_chunk_ids = [
        str(citation.get("chunk_id"))
        for citation in item.get("required_citations", [])
        if citation.get("chunk_id") is not None
    ]

    if any(value.startswith("seed:") for value in source_chunk_ids):
        reasons.append("source_chunk_ids")
    if any(value.startswith("seed:") for value in gold_chunk_ids):
        reasons.append("gold_chunk_ids")
    if any(value.startswith("seed:") for value in citation_chunk_ids):
        reasons.append("required_citations")

    expected_behavior = item.get("expected_behavior")
    expected_answer = item.get("expected_answer")
    if expected_behavior == "answer" and (expected_answer is None or not str(expected_answer).strip()):
        reasons.append("missing_expected_answer")

    return reasons


def is_bootstrap_item(item: dict) -> bool:
    return bool(bootstrap_reasons(item))


def is_live_eval_ready(item: dict) -> bool:
    return not bootstrap_reasons(item)


def readiness_label(item: dict) -> str:
    return "ready" if is_live_eval_ready(item) else "bootstrap"
