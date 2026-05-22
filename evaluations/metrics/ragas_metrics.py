"""Ragas integration helpers with graceful fallback when dependencies are unavailable."""

from __future__ import annotations

from collections.abc import Iterable


def _contexts_from_result(result: dict) -> list[str]:
    chunks = result.get("retrieved_chunks") or []
    contexts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "").strip()
        if text:
            contexts.append(text)
    return contexts


def build_records(items: list[dict], results: list[dict]) -> list[dict]:
    by_id = {item.get("id"): item for item in items}
    records: list[dict] = []
    for result in results:
        item = by_id.get(result.get("item_id")) or {}
        contexts = _contexts_from_result(result)
        records.append(
            {
                "item_id": result.get("item_id"),
                "question": result.get("question", ""),
                "answer": result.get("actual_answer", ""),
                "ground_truth": item.get("expected_answer") or "",
                "contexts": contexts,
                "eligible": bool(
                    result.get("status") == "completed"
                    and isinstance(item.get("expected_answer"), str)
                    and str(item.get("expected_answer") or "").strip()
                    and contexts
                    and str(result.get("actual_answer") or "").strip()
                ),
            }
        )
    return records


def _normalize_score_output(raw_result) -> dict[str, float | None]:
    metric_names = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]

    if hasattr(raw_result, "to_dict"):
        data = raw_result.to_dict()
        return {name: data.get(name) for name in metric_names}

    if isinstance(raw_result, dict):
        return {name: raw_result.get(name) for name in metric_names}

    normalized: dict[str, float | None] = {}
    for name in metric_names:
        normalized[name] = getattr(raw_result, name, None)
    return normalized


def evaluate_records(records: list[dict]) -> dict:
    eligible = [record for record in records if record.get("eligible")]
    if not eligible:
        return {
            "status": "skipped",
            "reason": "No eligible completed items with reference answers and retrieved contexts.",
            "eligible_records": 0,
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        return {
            "status": "unavailable",
            "reason": f"Ragas dependencies unavailable: {type(exc).__name__}: {exc}",
            "eligible_records": len(eligible),
        }

    dataset = Dataset.from_list(
        [
            {
                "question": record["question"],
                "answer": record["answer"],
                "ground_truth": record["ground_truth"],
                "contexts": record["contexts"],
            }
            for record in eligible
        ]
    )

    try:  # pragma: no cover - dependent on external package behavior
        raw_result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
                answer_correctness,
            ],
        )
        scores = _normalize_score_output(raw_result)
        return {
            "status": "ok",
            "eligible_records": len(eligible),
            "metrics": scores,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"Ragas evaluation failed: {type(exc).__name__}: {exc}",
            "eligible_records": len(eligible),
        }
