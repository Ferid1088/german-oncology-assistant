"""Run evaluation items against the live chat API and store results.

Usage:
    python evaluations/scripts/run_eval.py --api-key <key>
    python evaluations/scripts/run_eval.py --api-key <key> --split eval --limit 5
    python evaluations/scripts/run_eval.py --api-key <key> --api-url http://localhost:8000
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "eval" / "evaluation-dataset.json"
RESULTS_DIR = ROOT / "evaluations" / "results"

sys.path.insert(0, str(ROOT))
from evaluations.metrics import advanced, behavioral, ragas_metrics, retrieval, similarity
from evaluations.utils import (
    bootstrap_reasons,
    is_live_eval_ready,
    load_json_items,
    readiness_label,
    resolve_dataset_path,
)


def load_dataset(path: Path, split: str, ready_only: bool = False) -> list[dict]:
    items = load_json_items(path)
    selected = [i for i in items if i.get("dataset_split", "eval") == split]
    if split == "ab_test" and not selected:
        selected = [i for i in items if i.get("dataset_split", "eval") == "eval"]
    if ready_only:
        selected = [i for i in selected if is_live_eval_ready(i)]
    return selected


def _parse_sse(raw: str) -> dict:
    last_payload = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:") and "[DONE]" not in line:
            payload = line[5:].strip()
            try:
                last_payload = json.loads(payload)
            except json.JSONDecodeError:
                continue
    return last_payload


def call_api(
    item: dict,
    api_url: str,
    api_key: str,
    timeout: float = 120.0,
    max_retries: int = 1,
    retry_delay: float = 2.0,
) -> tuple[dict, float]:
    filters = item.get("expected_filters") or {}
    guideline_id = filters.get("guideline_id", "") if isinstance(filters, dict) else ""

    body = {
        "query": item["question"],
        "session_id": f"eval-{item['id']}",
        "guideline_id": guideline_id or "",
        "grade": "",
    }
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    started = time.perf_counter()
    timeout_config = httpx.Timeout(timeout, connect=min(timeout, 10.0))
    total_attempts = max(1, max_retries + 1)

    for attempt in range(1, total_attempts + 1):
        try:
            resp = httpx.post(
                f"{api_url.rstrip('/')}/chat",
                json=body,
                headers=headers,
                timeout=timeout_config,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            resp.raise_for_status()
            parsed = _parse_sse(resp.text)
            if not parsed:
                return {
                    "_error": "empty or invalid SSE payload",
                    "_error_type": "invalid_sse_payload",
                    "_attempts": attempt,
                    "_trace_id": resp.headers.get("X-Trace-Id"),
                }, elapsed_ms

            parsed.setdefault("_eval", {})
            parsed["_eval"].update({
                "attempts": attempt,
                "http_status": resp.status_code,
                "trace_id": resp.headers.get("X-Trace-Id") or parsed.get("trace_id"),
            })
            return parsed, elapsed_ms
        except httpx.HTTPStatusError as e:
            elapsed_ms = (time.perf_counter() - started) * 1000
            status_code = e.response.status_code
            if status_code >= 500 and attempt < total_attempts:
                time.sleep(retry_delay * attempt)
                continue
            return {
                "_error": f"HTTP {status_code}: {e.response.text[:200]}",
                "_error_type": "http_status_error",
                "_http_status": status_code,
                "_attempts": attempt,
                "_trace_id": e.response.headers.get("X-Trace-Id"),
            }, elapsed_ms
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if attempt < total_attempts:
                time.sleep(retry_delay * attempt)
                continue
            return {
                "_error": str(e) or type(e).__name__,
                "_error_type": type(e).__name__,
                "_attempts": attempt,
            }, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return {
                "_error": str(e),
                "_error_type": type(e).__name__,
                "_attempts": attempt,
            }, elapsed_ms

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {"_error": "unknown evaluation error", "_error_type": "unknown", "_attempts": total_attempts}, elapsed_ms


def evaluate_item(item: dict, resp: dict, elapsed_ms: float) -> dict:
    error = resp.get("_error")
    if error:
        return {
            "item_id": item["id"],
            "question": item["question"],
            "question_type": item.get("question_type"),
            "difficulty": item.get("difficulty"),
            "dataset_readiness": readiness_label(item),
            "bootstrap_reasons": bootstrap_reasons(item),
            "status": "error",
            "error": error,
            "error_type": resp.get("_error_type"),
            "attempts": resp.get("_attempts", 1),
            "trace_id": resp.get("_trace_id"),
            "response_time_ms": round(elapsed_ms, 1),
        }

    beh = behavioral.compute_all(item, resp)
    adv = advanced.compute_all(item, resp)
    ret = retrieval.compute_all(item, resp)

    expected_answer = item.get("expected_answer")
    if isinstance(expected_answer, str) and expected_answer.strip():
        sim = similarity.compute_all(
            expected_answer,
            resp.get("answer_professional", ""),
        )
    else:
        sim = {"token_overlap_f1": None, "char_overlap": None}

    token_usage = resp.get("token_usage", {})
    eval_meta = resp.get("_eval", {})

    return {
        "item_id": item["id"],
        "question": item["question"],
        "question_type": item.get("question_type"),
        "difficulty": item.get("difficulty"),
        "dataset_readiness": readiness_label(item),
        "bootstrap_reasons": bootstrap_reasons(item),
        "guideline_scope": item.get("guideline_scope"),
        "requires_clarification": item.get("requires_clarification", False),
        "expected_behavior": item.get("expected_behavior"),
        "status": "completed",
        "expected_answer": item.get("expected_answer", ""),
        "actual_answer": resp.get("answer_professional", ""),
        "actual_answer_plain": resp.get("answer_plain", ""),
        "response_time_ms": round(elapsed_ms, 1),
        "attempts": eval_meta.get("attempts", 1),
        "trace_id": eval_meta.get("trace_id") or resp.get("trace_id"),
        "token_usage": token_usage,
        "metrics": {**beh, **adv, **ret, **sim},
        "retrieved_chunks": resp.get("retrieved_chunks", []),
        "api_requires_clarification": resp.get("requires_clarification", False),
        "api_blocked": resp.get("blocked", False),
        "api_citations_count": len(resp.get("citations", [])),
        "api_tool_calls": [tc.get("tool") or tc.get("name", "") for tc in resp.get("tool_calls", [])],
        "safety_warning": resp.get("safety_warning", ""),
    }


def _safe_mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def _safe_rate(values: list[bool | None]) -> float | None:
    bools = [v for v in values if v is not None]
    return round(sum(bools) / len(bools), 4) if bools else None


def build_summary(run_id: str, split: str, results: list[dict], api_url: str, ragas_summary: dict | None = None) -> dict:
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "error"]

    def _metric(key: str):
        return [r["metrics"].get(key) for r in completed]

    by_type: dict[str, dict] = {}
    by_diff: dict[str, dict] = {}
    for r in completed:
        qt = r.get("question_type", "unknown")
        diff = r.get("difficulty", "unknown")
        for group, key in [(by_type, qt), (by_diff, diff)]:
            if key not in group:
                group[key] = {"count": 0, "behavioral_match": [], "token_overlap": []}
            group[key]["count"] += 1
            group[key]["behavioral_match"].append(r["metrics"].get("behavioral_match"))
            group[key]["token_overlap"].append(r["metrics"].get("token_overlap_f1"))

    def _agg(group: dict) -> dict:
        return {
            k: {
                "count": v["count"],
                "behavioral_match_rate": _safe_rate(v["behavioral_match"]),
                "token_overlap_mean": _safe_mean(v["token_overlap"]),
            }
            for k, v in group.items()
        }

    total_tokens = sum(
        (r.get("token_usage") or {}).get("total_tokens", 0)
        for r in completed
    )

    readiness_counts: dict[str, int] = {}
    error_types: dict[str, int] = {}
    for result in results:
        label = result.get("dataset_readiness", "unknown")
        readiness_counts[label] = readiness_counts.get(label, 0) + 1
        if result.get("status") == "error":
            err = result.get("error_type") or "unknown"
            error_types[err] = error_types.get(err, 0) + 1

    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "split": split,
        "total_items": len(results),
        "completed": len(completed),
        "failed": len(failed),
        "dataset_readiness": readiness_counts,
        "error_types": error_types,
        "metrics": {
            "behavioral_match_rate": _safe_rate(_metric("behavioral_match")),
            "clarification_correct_rate": _safe_rate(_metric("clarification_correct")),
            "citation_coverage_mean": _safe_mean(_metric("citation_coverage")),
            "tool_usage_match_rate": _safe_rate(_metric("tool_usage_match")),
            "blocked_correct_rate": _safe_rate(_metric("blocked_correct")),
            "answer_presence_rate": _safe_rate(_metric("answer_presence")),
            "citation_presence_rate": _safe_rate(_metric("citation_presence")),
            "plain_language_present_rate": _safe_rate(_metric("plain_language_present")),
            "expected_sections_coverage_mean": _safe_mean(_metric("expected_sections_coverage")),
            "recommendation_metadata_match_mean": _safe_mean(_metric("recommendation_metadata_match")),
            "claim_verdict_correct_rate": _safe_rate(_metric("claim_verdict_correct")),
            "retrieved_chunk_recall_mean": _safe_mean(_metric("retrieved_chunk_recall")),
            "retrieved_chunk_precision_mean": _safe_mean(_metric("retrieved_chunk_precision")),
            "top_gold_chunk_hit_rate": _safe_rate(_metric("top_gold_chunk_hit")),
            "retrieved_section_recall_mean": _safe_mean(_metric("retrieved_section_recall")),
            "retrieved_page_recall_mean": _safe_mean(_metric("retrieved_page_recall")),
            "citation_chunk_recall_mean": _safe_mean(_metric("citation_chunk_recall")),
            "answer_token_overlap_mean": _safe_mean(_metric("token_overlap_f1")),
            "avg_response_time_ms": _safe_mean([r["response_time_ms"] for r in completed]),
            "total_tokens_used": total_tokens,
        },
        "ragas": ragas_summary or {"status": "skipped", "reason": "Ragas was not evaluated."},
        "by_question_type": _agg(by_type),
        "by_difficulty": _agg(by_diff),
    }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation against the chat API.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--split", default="eval", choices=["eval", "guardrail", "smoke", "ab_test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--throttle-seconds", type=float, default=2.0)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=str(RESULTS_DIR))
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(ROOT, args.split, args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    items = load_dataset(dataset_path, args.split, ready_only=args.ready_only)
    if not items:
        readiness_hint = " with --ready-only" if args.ready_only else ""
        print(f"No items found for split='{args.split}'{readiness_hint}.", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        items = items[: args.limit]

    ready_count = sum(1 for item in items if is_live_eval_ready(item))
    bootstrap_count = len(items) - ready_count

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "api_url": args.api_url,
        "split": args.split,
        "dataset": str(dataset_path),
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
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"Run {run_id} | {len(items)} items | split={args.split}")
    print(f"Dataset readiness | ready={ready_count} | bootstrap={bootstrap_count} | ready_only={args.ready_only}")
    print(f"Output: {out_dir}\n")

    all_results = []
    for idx, item in enumerate(items, 1):
        if idx > 1:
            time.sleep(args.throttle_seconds)
        print(f"[{idx}/{len(items)}] {item['id']} ({item.get('question_type')}, {readiness_label(item)}) ... ", end="", flush=True)
        resp, elapsed_ms = call_api(
            item,
            args.api_url,
            args.api_key,
            timeout=args.timeout,
            max_retries=args.retries,
            retry_delay=args.retry_delay,
        )
        result = evaluate_item(item, resp, elapsed_ms)
        all_results.append(result)
        status = result["status"]
        bm = result.get("metrics", {}).get("behavioral_match")
        tok = result.get("metrics", {}).get("token_overlap_f1")
        tok_str = f"{tok:.2f}" if tok is not None else "-"
        bm_str = "✓" if bm is True else ("✗" if bm is False else "-")
        attempts = result.get("attempts", 1)
        print(f"{status} | {elapsed_ms:.0f}ms | attempts={attempts} | behavioral={bm_str} | sim={tok_str}")

    ragas_records = ragas_metrics.build_records(items, all_results)
    ragas_summary = ragas_metrics.evaluate_records(ragas_records)
    summary = build_summary(run_id, args.split, all_results, args.api_url, ragas_summary=ragas_summary)

    (out_dir / "item_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )
    (out_dir / "ragas_records.json").write_text(
        json.dumps(ragas_records, indent=2, ensure_ascii=False)
    )
    (out_dir / "ragas_summary.json").write_text(
        json.dumps(ragas_summary, indent=2, ensure_ascii=False)
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(f"\nCompleted: {summary['completed']}/{summary['total_items']}")
    m = summary["metrics"]
    print(f"Behavioral match rate: {m['behavioral_match_rate']}")
    print(f"Token overlap mean:    {m['answer_token_overlap_mean']}")
    print(f"Avg response time:     {m['avg_response_time_ms']} ms")
    print(f"Ragas status:          {summary['ragas'].get('status')}")
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
