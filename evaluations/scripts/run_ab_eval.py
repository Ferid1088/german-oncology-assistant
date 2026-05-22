"""A/B evaluation runner — same questions sent to two API variants, results compared.

Variant A = baseline (e.g. hybrid + reranker)
Variant B = challenger (e.g. hybrid without reranker, different model, etc.)

Usage:
    python evaluations/scripts/run_ab_eval.py \\
        --variant-a-url http://localhost:8000 \\
        --variant-b-url http://localhost:8001 \\
        --api-key <key>

    # Different keys per variant:
    python evaluations/scripts/run_ab_eval.py \\
        --variant-a-url http://localhost:8000 --api-key-a <key-a> \\
        --variant-b-url http://localhost:8001 --api-key-b <key-b>

    # Limit to first 5 items:
    python evaluations/scripts/run_ab_eval.py \\
        --variant-a-url http://localhost:8000 \\
        --variant-b-url http://localhost:8001 \\
        --api-key <key> --limit 5
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "eval" / "evaluation-dataset.json"
RESULTS_DIR = ROOT / "evaluations" / "results"

sys.path.insert(0, str(ROOT))
from evaluations.metrics import ragas_metrics
from evaluations.scripts.run_eval import load_dataset, call_api, evaluate_item
from evaluations.utils import is_live_eval_ready, readiness_label, resolve_dataset_path


def _safe_mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def _safe_rate(values: list[bool | None]) -> float | None:
    bools = [v for v in values if v is not None]
    return round(sum(bools) / len(bools), 4) if bools else None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _run_variant(
    items: list[dict],
    api_url: str,
    api_key: str,
    label: str,
    timeout: float,
    retries: int,
    retry_delay: float,
    throttle_seconds: float,
) -> list[dict]:
    results = []
    for idx, item in enumerate(items, 1):
        if idx > 1:
            time.sleep(throttle_seconds)
        print(f"  [{label}] [{idx}/{len(items)}] {item['id']} ({readiness_label(item)}) ... ", end="", flush=True)
        resp, elapsed_ms = call_api(
            item,
            api_url,
            api_key,
            timeout=timeout,
            max_retries=retries,
            retry_delay=retry_delay,
        )
        result = evaluate_item(item, resp, elapsed_ms)
        results.append(result)
        bm = result.get("metrics", {}).get("behavioral_match")
        tok = result.get("metrics", {}).get("token_overlap_f1")
        attempts = result.get("attempts", 1)
        print(
            f"{result['status']} | {elapsed_ms:.0f}ms | "
            f"attempts={attempts} | "
            f"behavioral={'✓' if bm else '✗' if bm is False else '-'} | "
            f"sim={f'{tok:.2f}' if tok is not None else '-'}"
        )
    return results


def _aggregate_metrics(results: list[dict]) -> dict:
    completed = [r for r in results if r["status"] == "completed"]

    def _m(key):
        return [r["metrics"].get(key) for r in completed]

    return {
        "behavioral_match_rate": _safe_rate(_m("behavioral_match")),
        "clarification_correct_rate": _safe_rate(_m("clarification_correct")),
        "citation_coverage_mean": _safe_mean(_m("citation_coverage")),
        "tool_usage_match_rate": _safe_rate(_m("tool_usage_match")),
        "answer_presence_rate": _safe_rate(_m("answer_presence")),
        "citation_presence_rate": _safe_rate(_m("citation_presence")),
        "plain_language_present_rate": _safe_rate(_m("plain_language_present")),
        "expected_sections_coverage_mean": _safe_mean(_m("expected_sections_coverage")),
        "recommendation_metadata_match_mean": _safe_mean(_m("recommendation_metadata_match")),
        "claim_verdict_correct_rate": _safe_rate(_m("claim_verdict_correct")),
        "retrieved_chunk_recall_mean": _safe_mean(_m("retrieved_chunk_recall")),
        "retrieved_chunk_precision_mean": _safe_mean(_m("retrieved_chunk_precision")),
        "top_gold_chunk_hit_rate": _safe_rate(_m("top_gold_chunk_hit")),
        "retrieved_section_recall_mean": _safe_mean(_m("retrieved_section_recall")),
        "retrieved_page_recall_mean": _safe_mean(_m("retrieved_page_recall")),
        "citation_chunk_recall_mean": _safe_mean(_m("citation_chunk_recall")),
        "answer_token_overlap_mean": _safe_mean(_m("token_overlap_f1")),
        "avg_response_time_ms": _safe_mean([r["response_time_ms"] for r in completed]),
        "completed": len(completed),
        "failed": len(results) - len(completed),
    }


def _aggregate_by_type(results: list[dict]) -> dict:
    by_type: dict[str, dict] = {}
    for r in results:
        if r["status"] != "completed":
            continue
        qt = r.get("question_type", "unknown")
        if qt not in by_type:
            by_type[qt] = {"behavioral_match": [], "token_overlap": [], "response_time": []}
        by_type[qt]["behavioral_match"].append(r["metrics"].get("behavioral_match"))
        by_type[qt]["token_overlap"].append(r["metrics"].get("token_overlap_f1"))
        by_type[qt]["response_time"].append(r["response_time_ms"])
    return {
        qt: {
            "behavioral_match_rate": _safe_rate(v["behavioral_match"]),
            "token_overlap_mean": _safe_mean(v["token_overlap"]),
            "avg_response_time_ms": _safe_mean(v["response_time"]),
            "count": len(v["behavioral_match"]),
        }
        for qt, v in by_type.items()
    }


def build_ab_item_results(
    items: list[dict],
    results_a: list[dict],
    results_b: list[dict],
) -> list[dict]:
    out = []
    for item, ra, rb in zip(items, results_a, results_b):
        ma = ra.get("metrics", {})
        mb = rb.get("metrics", {})

        delta_metrics = {
            k: _delta(ma.get(k), mb.get(k))
            for k in [
                "token_overlap_f1",
                "char_overlap",
                "citation_coverage",
                "expected_sections_coverage",
                "recommendation_metadata_match",
            ]
        }
        delta_metrics["response_time_ms"] = _delta(
            ra.get("response_time_ms"), rb.get("response_time_ms")
        )

        out.append({
            "item_id": item["id"],
            "question": item["question"],
            "question_type": item.get("question_type"),
            "difficulty": item.get("difficulty"),
            "expected_answer": item.get("expected_answer", ""),
            "dataset_readiness": readiness_label(item),
            "variant_a": {
                "status": ra["status"],
                "answer": ra.get("actual_answer", ""),
                "metrics": ma,
                "response_time_ms": ra.get("response_time_ms"),
                "attempts": ra.get("attempts", 1),
                "trace_id": ra.get("trace_id"),
                "api_blocked": ra.get("api_blocked"),
                "api_requires_clarification": ra.get("api_requires_clarification"),
                "api_citations_count": ra.get("api_citations_count", 0),
                "error": ra.get("error"),
            },
            "variant_b": {
                "status": rb["status"],
                "answer": rb.get("actual_answer", ""),
                "metrics": mb,
                "response_time_ms": rb.get("response_time_ms"),
                "attempts": rb.get("attempts", 1),
                "trace_id": rb.get("trace_id"),
                "api_blocked": rb.get("api_blocked"),
                "api_requires_clarification": rb.get("api_requires_clarification"),
                "api_citations_count": rb.get("api_citations_count", 0),
                "error": rb.get("error"),
            },
            "delta_b_minus_a": delta_metrics,
            "winner": _pick_winner(ma, mb, ra, rb),
        })
    return out


def _pick_winner(ma: dict, mb: dict, ra: dict, rb: dict) -> str:
    """Weighted heuristic balancing behavior, grounding, structure, and latency."""
    if ra.get("status") != "completed" and rb.get("status") == "completed":
        return "B"
    if rb.get("status") != "completed" and ra.get("status") == "completed":
        return "A"
    if ra.get("status") != "completed" and rb.get("status") != "completed":
        return "tie"

    def _score(metrics: dict, result: dict) -> float:
        score = 0.0
        if metrics.get("behavioral_match") is True:
            score += 3.0
        if metrics.get("clarification_correct") is True:
            score += 1.5
        if metrics.get("citation_presence") is True:
            score += 1.5
        if metrics.get("plain_language_present") is True:
            score += 0.5
        score += float(metrics.get("citation_coverage") or 0.0) * 1.5
        score += float(metrics.get("retrieved_chunk_recall") or 0.0) * 1.5
        score += float(metrics.get("retrieved_section_recall") or 0.0) * 1.0
        score += float(metrics.get("retrieved_page_recall") or 0.0) * 1.0
        score += float(metrics.get("expected_sections_coverage") or 0.0) * 1.5
        score += float(metrics.get("recommendation_metadata_match") or 0.0) * 1.0
        score += float(metrics.get("token_overlap_f1") or 0.0) * 2.0
        score -= float(result.get("response_time_ms") or 0.0) / 10000.0
        return round(score, 4)

    score_a = _score(ma, ra)
    score_b = _score(mb, rb)
    if score_b - score_a > 0.2:
        return "B"
    if score_a - score_b > 0.2:
        return "A"
    return "tie"


def build_ab_summary(
    run_id: str,
    split: str,
    variant_a_url: str,
    variant_b_url: str,
    results_a: list[dict],
    results_b: list[dict],
    ab_items: list[dict],
    ragas_a: dict | None = None,
    ragas_b: dict | None = None,
) -> dict:
    agg_a = _aggregate_metrics(results_a)
    agg_b = _aggregate_metrics(results_b)
    by_type_a = _aggregate_by_type(results_a)
    by_type_b = _aggregate_by_type(results_b)

    delta_global = {
        k: _delta(agg_a.get(k), agg_b.get(k))
        for k in [
            "behavioral_match_rate", "clarification_correct_rate",
            "citation_coverage_mean", "answer_presence_rate",
            "citation_presence_rate", "expected_sections_coverage_mean",
            "recommendation_metadata_match_mean", "answer_token_overlap_mean",
            "retrieved_chunk_recall_mean", "retrieved_chunk_precision_mean",
            "top_gold_chunk_hit_rate", "retrieved_section_recall_mean",
            "retrieved_page_recall_mean", "citation_chunk_recall_mean",
            "avg_response_time_ms",
        ]
    }

    ragas_delta = {}
    if (ragas_a or {}).get("status") == "ok" and (ragas_b or {}).get("status") == "ok":
        for key in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
            ragas_delta[key] = _delta(
                ((ragas_a or {}).get("metrics") or {}).get(key),
                ((ragas_b or {}).get("metrics") or {}).get(key),
            )

    winners = [i["winner"] for i in ab_items]
    winner_counts = {
        "A": winners.count("A"),
        "B": winners.count("B"),
        "tie": winners.count("tie"),
    }

    by_type_comparison = {}
    for qt in set(list(by_type_a.keys()) + list(by_type_b.keys())):
        ta = by_type_a.get(qt, {})
        tb = by_type_b.get(qt, {})
        by_type_comparison[qt] = {
            "variant_a": ta,
            "variant_b": tb,
            "delta_token_overlap": _delta(ta.get("token_overlap_mean"), tb.get("token_overlap_mean")),
            "delta_behavioral_match": _delta(ta.get("behavioral_match_rate"), tb.get("behavioral_match_rate")),
            "delta_response_time_ms": _delta(ta.get("avg_response_time_ms"), tb.get("avg_response_time_ms")),
        }

    overall_winner = "tie"
    max_count = max(winner_counts.values()) if winner_counts else 0
    leaders = [key for key, value in winner_counts.items() if value == max_count]
    if len(leaders) == 1:
        overall_winner = leaders[0]

    readiness_counts = {
        "ready": sum(1 for item in ab_items if item.get("dataset_readiness") == "ready"),
        "bootstrap": sum(1 for item in ab_items if item.get("dataset_readiness") == "bootstrap"),
    }

    return {
        "run_id": run_id,
        "run_type": "ab_eval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "variant_a_url": variant_a_url,
        "variant_b_url": variant_b_url,
        "total_items": len(results_a),
        "dataset_readiness": readiness_counts,
        "winner_counts": winner_counts,
        "overall_winner": overall_winner,
        "metrics": {
            "variant_a": agg_a,
            "variant_b": agg_b,
            "delta_b_minus_a": delta_global,
        },
        "ragas": {
            "variant_a": ragas_a or {"status": "skipped"},
            "variant_b": ragas_b or {"status": "skipped"},
            "delta_b_minus_a": ragas_delta,
        },
        "by_question_type": by_type_comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="A/B evaluation runner.")
    parser.add_argument("--variant-a-url", required=True, help="API URL for variant A (baseline)")
    parser.add_argument("--variant-b-url", required=True, help="API URL for variant B (challenger)")
    parser.add_argument("--api-key", default=None, help="Shared API key for both variants")
    parser.add_argument("--api-key-a", default=None, help="API key for variant A (overrides --api-key)")
    parser.add_argument("--api-key-b", default=None, help="API key for variant B (overrides --api-key)")
    parser.add_argument("--split", default="eval", choices=["eval", "ab_test", "guardrail", "smoke"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--throttle-seconds", type=float, default=2.0)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=str(RESULTS_DIR))
    args = parser.parse_args()

    key_a = args.api_key_a or args.api_key
    key_b = args.api_key_b or args.api_key
    if not key_a or not key_b:
        print("ERROR: provide --api-key (shared) or --api-key-a / --api-key-b", file=sys.stderr)
        sys.exit(1)

    dataset_path = resolve_dataset_path(ROOT, args.split, args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    items = load_dataset(dataset_path, args.split, ready_only=args.ready_only)
    if not items:
        readiness_hint = " with --ready-only" if args.ready_only else ""
        print(f"No items for split='{args.split}'{readiness_hint}. Use --split eval to run on main dataset.", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        items = items[: args.limit]

    ready_count = sum(1 for item in items if is_live_eval_ready(item))
    bootstrap_count = len(items) - ready_count

    run_id = f"ab_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"A/B Run {run_id} | {len(items)} items | split={args.split}")
    print(f"Variant A: {args.variant_a_url}")
    print(f"Variant B: {args.variant_b_url}")
    print(f"Dataset readiness | ready={ready_count} | bootstrap={bootstrap_count} | ready_only={args.ready_only}\n")

    print("── Running Variant A ──────────────────────────────")
    results_a = _run_variant(items, args.variant_a_url, key_a, "A", args.timeout, args.retries, args.retry_delay, args.throttle_seconds)

    print("\n── Running Variant B ──────────────────────────────")
    results_b = _run_variant(items, args.variant_b_url, key_b, "B", args.timeout, args.retries, args.retry_delay, args.throttle_seconds)

    print("\n── Building comparison ────────────────────────────")
    ab_items = build_ab_item_results(items, results_a, results_b)
    ragas_records_a = ragas_metrics.build_records(items, results_a)
    ragas_records_b = ragas_metrics.build_records(items, results_b)
    ragas_a = ragas_metrics.evaluate_records(ragas_records_a)
    ragas_b = ragas_metrics.evaluate_records(ragas_records_b)
    summary = build_ab_summary(
        run_id, args.split,
        args.variant_a_url, args.variant_b_url,
        results_a, results_b, ab_items,
        ragas_a=ragas_a,
        ragas_b=ragas_b,
    )

    (out_dir / "metadata.json").write_text(json.dumps({
        "run_id": run_id,
        "run_type": "ab_eval",
        "variant_a_url": args.variant_a_url,
        "variant_b_url": args.variant_b_url,
        "split": args.split,
        "total_items": len(items),
        "ready_only": args.ready_only,
        "ready_items": ready_count,
        "bootstrap_items": bootstrap_count,
        "limit": args.limit,
        "timeout_seconds": args.timeout,
        "retries": args.retries,
        "retry_delay_seconds": args.retry_delay,
        "throttle_seconds": args.throttle_seconds,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False))
    (out_dir / "ab_item_results.json").write_text(
        json.dumps(ab_items, indent=2, ensure_ascii=False)
    )
    (out_dir / "ragas_records_variant_a.json").write_text(
        json.dumps(ragas_records_a, indent=2, ensure_ascii=False)
    )
    (out_dir / "ragas_records_variant_b.json").write_text(
        json.dumps(ragas_records_b, indent=2, ensure_ascii=False)
    )
    (out_dir / "ab_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    wc = summary["winner_counts"]
    m = summary["metrics"]
    print(f"\nWinner counts — A: {wc['A']}  B: {wc['B']}  Tie: {wc['tie']}")
    print(f"Overall winner: {summary['overall_winner']}")
    print(f"\nToken overlap — A: {m['variant_a'].get('answer_token_overlap_mean')}  "
          f"B: {m['variant_b'].get('answer_token_overlap_mean')}  "
          f"Δ: {m['delta_b_minus_a'].get('answer_token_overlap_mean')}")
    print(f"Avg latency   — A: {m['variant_a'].get('avg_response_time_ms')} ms  "
          f"B: {m['variant_b'].get('avg_response_time_ms')} ms  "
          f"Δ: {m['delta_b_minus_a'].get('avg_response_time_ms')} ms")
    print(f"Ragas        — A: {summary['ragas']['variant_a'].get('status')}  B: {summary['ragas']['variant_b'].get('status')}")
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
