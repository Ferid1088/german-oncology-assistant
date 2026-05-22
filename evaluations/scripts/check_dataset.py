"""Audit the evaluation dataset against the JSON schema and documented strategy.

Usage:
    python evaluations/scripts/check_dataset.py
    python evaluations/scripts/check_dataset.py --dataset data/eval/evaluation-dataset.json
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

DEFAULT_DATASET = ROOT / "data" / "eval" / "evaluation-dataset.json"
DEFAULT_SCHEMA = ROOT / "docs" / "evaluation-dataset.schema.json"
RESULTS_DIR = ROOT / "evaluations" / "results"
from evaluations.utils import bootstrap_reasons, discover_dataset_paths, load_json_items, resolve_dataset_path


TARGET_DISTRIBUTION = {
    "recommendation": 10,
    "factual": 6,
    "evidence": 5,
    "comparison": 4,
    "drug_lookup": 3,
    "external": 2,
}

EXPECTED_SPLITS = {"eval", "ab_test", "guardrail", "smoke"}
REQUIRED_FIELDS = [
    "id", "question", "question_type", "difficulty", "guideline_scope",
    "expected_answer", "gold_chunk_ids", "source_chunk_ids",
    "dataset_split", "expected_behavior", "expected_answer_format",
]


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_split_items(dataset_path: Path) -> tuple[dict[str, list[dict]], list[dict]]:
    discovered = discover_dataset_paths(dataset_path)
    split_items: dict[str, list[dict]] = {}

    for split_name, path in discovered.items():
        if split_name == "ab_test" and path == discovered.get("eval"):
            continue
        split_items[split_name] = load_json_items(path)

    if "eval" in split_items:
        split_items.setdefault(
            "ab_test",
            [{**item, "dataset_split": "ab_test"} for item in split_items["eval"]],
        )

    all_physical_items: list[dict] = []
    for split_name, items in split_items.items():
        if split_name == "ab_test":
            continue
        all_physical_items.extend(items)

    return split_items, all_physical_items


def check_schema(items: list[dict], schema: dict) -> list[str]:
    if not _HAS_JSONSCHEMA:
        return ["jsonschema not installed — schema validation skipped (pip install jsonschema)"]
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    item_schema = schema.get("items", schema)
    for item in items:
        for err in jsonschema.Draft202012Validator(item_schema).iter_errors(item):
            errors.append(f"{item.get('id', '?')}: {err.message}")
    return errors


def check_distribution(items: list[dict]) -> tuple[dict, bool]:
    actual: dict[str, int] = {}
    for item in items:
        qt = item.get("question_type", "unknown")
        actual[qt] = actual.get(qt, 0) + 1
    match = all(actual.get(k, 0) == v for k, v in TARGET_DISTRIBUTION.items())
    return actual, match


def check_splits(items: list[dict]) -> list[str]:
    present = {item.get("dataset_split") for item in items}
    return sorted(EXPECTED_SPLITS - present)


def check_bootstrap(items: list[dict]) -> tuple[int, int]:
    bootstrap = 0
    for item in items:
        if bootstrap_reasons(item):
            bootstrap += 1
    ready = len(items) - bootstrap
    return bootstrap, ready


def check_readiness(items: list[dict]) -> tuple[dict, dict, list[dict]]:
    by_split: dict[str, dict] = {}
    by_reason: dict[str, int] = {}
    examples: list[dict] = []

    for item in items:
        split = item.get("dataset_split", "unknown")
        entry = by_split.setdefault(split, {"total": 0, "ready": 0, "bootstrap": 0})
        entry["total"] += 1

        reasons = bootstrap_reasons(item)
        if reasons:
            entry["bootstrap"] += 1
            for reason in reasons:
                by_reason[reason] = by_reason.get(reason, 0) + 1
            if len(examples) < 10:
                examples.append({
                    "id": item.get("id", "?"),
                    "dataset_split": split,
                    "reasons": reasons,
                })
        else:
            entry["ready"] += 1

    return by_split, by_reason, examples


def check_required_fields(items: list[dict]) -> list[str]:
    issues = []
    for item in items:
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        if missing:
            issues.append(f"{item.get('id', '?')}: missing {missing}")
    return issues


def run(dataset_path: Path | str, schema_path: Path | str, split: str | None = None) -> dict:
    dataset_path = Path(dataset_path)
    schema_path = Path(schema_path)
    split_items, items = load_split_items(dataset_path)
    schema = load_json(schema_path)

    schema_errors = check_schema(items, schema)
    distribution, dist_match = check_distribution(split_items.get("eval", []))
    missing_splits = sorted(EXPECTED_SPLITS - set(split_items.keys()))
    bootstrap_count, ready_count = check_bootstrap(items)
    readiness_by_split, bootstrap_reason_counts, bootstrap_examples = check_readiness(
        [item for values in split_items.values() for item in values]
    )
    field_issues = check_required_fields(items)

    selected_items = list(split_items.get(split, [])) if split else list(items)
    selected_bootstrap, selected_ready = check_bootstrap(selected_items)
    split_exists = bool(selected_items) if split else True

    schema_valid = len(schema_errors) == 0
    blocking_ok = schema_valid and not field_issues and split_exists
    readiness_ok = (selected_ready == len(selected_items)) if split else (ready_count == len(items) if items else False)
    overall_ok = blocking_ok and dist_match and readiness_ok and not missing_splits

    clarification_items = [i for i in split_items.get("eval", []) if i.get("requires_clarification")]
    rec_with_metadata = [
        i for i in split_items.get("eval", [])
        if i.get("question_type") == "recommendation" and i.get("expected_recommendation_metadata")
    ]

    result = {
        "timestamp": date.today().isoformat(),
        "dataset_path": str(dataset_path),
        "schema_valid": schema_valid,
        "schema_errors": schema_errors[:20],
        "total_items": len(items),
        "distribution": distribution,
        "target_distribution": TARGET_DISTRIBUTION,
        "distribution_match": dist_match,
        "missing_splits": missing_splits,
        "bootstrap_items": bootstrap_count,
        "items_ready_for_eval": ready_count,
        "readiness_by_split": readiness_by_split,
        "bootstrap_reason_counts": bootstrap_reason_counts,
        "bootstrap_examples": bootstrap_examples,
        "selected_split": split,
        "selected_split_total_items": len(selected_items),
        "selected_split_ready_items": selected_ready,
        "selected_split_bootstrap_items": selected_bootstrap,
        "clarification_items": len(clarification_items),
        "recommendation_items_with_metadata": len(rec_with_metadata),
        "field_issues": field_issues[:20],
        "blocking_ok": blocking_ok,
        "readiness_ok": readiness_ok,
        "overall_ok": overall_ok,
        "summary": _make_summary(
            len(items), schema_valid, dist_match, missing_splits,
            bootstrap_count, ready_count, field_issues,
            split=split,
            split_exists=split_exists,
            selected_total=len(selected_items),
            selected_ready=selected_ready,
        ),
    }
    return result


def _make_summary(total, schema_valid, dist_match, missing_splits, bootstrap, ready, field_issues, *, split=None, split_exists=True, selected_total=0, selected_ready=0) -> str:
    parts = []
    parts.append(f"{total} items total.")
    parts.append("Schema: valid." if schema_valid else "Schema: INVALID — see schema_errors.")
    parts.append("Distribution: matches target." if dist_match else "Distribution: MISMATCH vs target.")
    if missing_splits:
        parts.append(f"Missing splits: {', '.join(missing_splits)} — these datasets do not exist yet.")
    parts.append(f"Bootstrap items (seed:* chunk IDs): {bootstrap}/{total}. Ready for live eval: {ready}.")
    if field_issues:
        parts.append(f"Field issues: {len(field_issues)} items have missing required fields.")
    if split:
        if split_exists:
            parts.append(f"Selected split '{split}': {selected_total} items, {selected_ready} live-ready.")
        else:
            parts.append(f"Selected split '{split}': no items found.")
    if ready < total:
        parts.append("Readiness: warnings present — bootstrap items still need manual curation before full live evaluation.")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Audit the evaluation dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--split", choices=["eval", "ab_test", "guardrail", "smoke"], default=None)
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(ROOT, args.split or "eval", args.dataset)
    schema_path = Path(args.schema)

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    if not schema_path.exists():
        print(f"ERROR: schema not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    result = run(dataset_path, schema_path, split=args.split)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"dataset_check_{date.today().isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(result["summary"])
    print(f"\nFull report saved to: {out_path}")
    sys.exit(0 if result["blocking_ok"] else 1)


if __name__ == "__main__":
    main()
