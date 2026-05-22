"""Evaluation dashboard for the oncology assistant.

Run with:
    streamlit run evaluations/ui/app.py
"""

from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import evaluations.utils as eval_utils


bootstrap_reasons = eval_utils.bootstrap_reasons
readiness_label = eval_utils.readiness_label


def _discover_dataset_paths(dataset_path: Path) -> dict[str, Path]:
    helper = getattr(eval_utils, "discover_dataset_paths", None)
    if callable(helper):
        return helper(dataset_path)

    base_dir = dataset_path.parent
    discovered: dict[str, Path] = {}
    filename_map = {
        "eval": "evaluation-dataset.json",
        "ab_test": "evaluation-dataset.json",
        "guardrail": "guardrail-dataset.json",
        "smoke": "smoke-dataset.json",
    }
    for split, filename in filename_map.items():
        candidate = base_dir / filename
        if candidate.exists():
            discovered[split] = candidate
    if "eval" in discovered:
        discovered.setdefault("ab_test", discovered["eval"])
    return discovered


def _load_json_items(path: Path) -> list[dict]:
    helper = getattr(eval_utils, "load_json_items", None)
    if callable(helper):
        return helper(path)
    payload = _load_json(path)
    if isinstance(payload, list):
        return payload
    return payload.get("items", []) if isinstance(payload, dict) else []


def _normalize_scope_labels(value: object) -> list[str]:
    helper = getattr(eval_utils, "normalize_scope_labels", None)
    if callable(helper):
        return helper(value)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value)]

RESULTS_DIR = ROOT / "evaluations" / "results"
DATASET_PATH = ROOT / "data" / "eval" / "evaluation-dataset.json"
SCHEMA_PATH = ROOT / "docs" / "evaluation-dataset.schema.json"

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="🔬",
    layout="wide",
)

_CSS = """
<style>
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1e293b; }
.metric-label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.pass { color: #16a34a; font-weight: 700; }
.fail { color: #dc2626; font-weight: 700; }
.warn { color: #d97706; font-weight: 700; }
.seed { background: #fef9c3; border-radius: 4px; padding: 1px 6px; font-size: 0.8rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.0f} ms"


def _load_runs() -> list[dict]:
    runs = []
    if not RESULTS_DIR.exists():
        return runs
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        summary_path = d / "summary.json"
        if d.is_dir() and summary_path.exists():
            summary = _load_json(summary_path)
            if summary:
                runs.append({"run_id": d.name, "path": d, "summary": summary})
    return runs


def _load_ab_runs() -> list[dict]:
    ab_runs = []
    if not RESULTS_DIR.exists():
        return ab_runs
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        ab_summary_path = d / "ab_summary.json"
        if d.is_dir() and ab_summary_path.exists():
            summary = _load_json(ab_summary_path)
            if summary:
                ab_runs.append({"run_id": d.name, "path": d, "summary": summary})
    return ab_runs


def _load_dataset() -> list[dict]:
    discovered = _discover_dataset_paths(DATASET_PATH)
    items: list[dict] = []
    for split_name, path in discovered.items():
        if split_name == "ab_test" and path == discovered.get("eval"):
            continue
        items.extend(_load_json_items(path))
    return items


def _latest_dataset_check() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    reports = sorted(
        [path for path in RESULTS_DIR.iterdir() if path.name.startswith("dataset_check_") and path.name.endswith(".json")],
        reverse=True,
    )
    return reports[0] if reports else None


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("🔬 Evaluation Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Dataset Overview", "Evaluation Runs", "Run Details", "Item Inspector", "A/B Comparison"],
)

runs = _load_runs()
ab_runs = _load_ab_runs()
dataset = _load_dataset()

# ── Dataset Overview ──────────────────────────────────────────────────────────

if page == "Dataset Overview":
    st.title("Dataset Overview")

    if not dataset:
        st.error(f"Dataset not found at `{DATASET_PATH}`")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    bootstrap = sum(1 for i in dataset if bootstrap_reasons(i))
    ready = len(dataset) - bootstrap
    clarification_items = sum(1 for i in dataset if i.get("requires_clarification"))
    splits_present = {i.get("dataset_split") for i in dataset}
    missing_splits = {"eval", "ab_test", "guardrail", "smoke"} - splits_present

    with col1:
        st.metric("Total Items", len(dataset))
    with col2:
        st.metric("Ready for Live Eval", ready)
    with col3:
        st.metric("Bootstrap (seed:*)", bootstrap)
    with col4:
        st.metric("Clarification Items", clarification_items)

    if missing_splits:
        st.warning(f"Missing dataset splits: **{', '.join(sorted(missing_splits))}** — these need to be created.")

    st.subheader("Distribution by Question Type")
    dist = Counter(i.get("question_type", "unknown") for i in dataset)
    target = {"recommendation": 10, "factual": 6, "evidence": 5, "comparison": 4, "drug_lookup": 3, "external": 2}

    dist_rows = []
    for qt, target_n in target.items():
        actual_n = dist.get(qt, 0)
        match = "✓" if actual_n == target_n else "✗"
        dist_rows.append({"Question Type": qt, "Target": target_n, "Actual": actual_n, "Match": match})
    for qt, n in dist.items():
        if qt not in target:
            dist_rows.append({"Question Type": qt, "Target": "—", "Actual": n, "Match": "—"})

    st.dataframe(dist_rows, width="stretch", hide_index=True)

    st.subheader("Distribution by Difficulty")
    diff_dist = Counter(i.get("difficulty", "unknown") for i in dataset)
    st.bar_chart(dict(diff_dist))

    st.subheader("Distribution by Guideline Scope")
    scope_dist = Counter()
    for item in dataset:
        labels = _normalize_scope_labels(item.get("guideline_scope")) or ["unknown"]
        scope_dist.update(labels)
    st.bar_chart(dict(scope_dist))

    st.subheader("Readiness by Split")
    readiness_rows = []
    for split in sorted(splits_present):
        items_in_split = [item for item in dataset if item.get("dataset_split") == split]
        split_ready = sum(1 for item in items_in_split if readiness_label(item) == "ready")
        split_bootstrap = len(items_in_split) - split_ready
        readiness_rows.append({
            "Split": split,
            "Total": len(items_in_split),
            "Ready": split_ready,
            "Bootstrap": split_bootstrap,
        })
    st.dataframe(readiness_rows, width="stretch", hide_index=True)

    reason_counts = Counter()
    for item in dataset:
        reason_counts.update(bootstrap_reasons(item))
    if reason_counts:
        st.subheader("Bootstrap Readiness Reasons")
        st.dataframe(
            [{"Reason": reason, "Count": count} for reason, count in reason_counts.most_common()],
            width="stretch",
            hide_index=True,
        )

    latest_check = _latest_dataset_check()

    if latest_check and latest_check.exists():
        st.subheader("Latest Dataset Check Report")
        check = _load_json(latest_check)
        if check:
            st.info(check.get("summary", ""))
            c1, c2, c3 = st.columns(3)
            c1.metric("Schema valid", "Yes" if check.get("schema_valid") else "No")
            c2.metric("Blocking OK", "Yes" if check.get("blocking_ok") else "No")
            c3.metric("Readiness OK", "Yes" if check.get("readiness_ok") else "No")
            if check.get("schema_errors"):
                st.error("Schema errors:\n" + "\n".join(check["schema_errors"][:10]))
            if check.get("field_issues"):
                st.warning("Field issues:\n" + "\n".join(check["field_issues"][:10]))
            if check.get("bootstrap_examples"):
                st.caption("Example bootstrap items")
                st.dataframe(check["bootstrap_examples"], width="stretch", hide_index=True)
    else:
        st.info("Run `python evaluations/scripts/check_dataset.py` to generate a dataset check report.")

# ── Evaluation Runs ───────────────────────────────────────────────────────────

elif page == "Evaluation Runs":
    st.title("Evaluation Runs")

    if not runs:
        st.info("No evaluation runs found. Run `python evaluations/scripts/run_eval.py --api-key <key>` to create one.")
        st.stop()

    rows = []
    for r in runs:
        s = r["summary"]
        m = s.get("metrics", {})
        rows.append({
            "Run ID": s.get("run_id", r["run_id"]),
            "Split": s.get("split", "—"),
            "Items": s.get("total_items", "—"),
            "Completed": s.get("completed", "—"),
            "Failed": s.get("failed", 0),
            "Ready": (s.get("dataset_readiness") or {}).get("ready", 0),
            "Bootstrap": (s.get("dataset_readiness") or {}).get("bootstrap", 0),
            "Behavioral Match": _pct(m.get("behavioral_match_rate")),
            "Citation Presence": _pct(m.get("citation_presence_rate")),
            "Token Overlap": _pct(m.get("answer_token_overlap_mean")),
            "Avg Latency": _ms(m.get("avg_response_time_ms")),
            "Timestamp": s.get("timestamp", "—")[:19].replace("T", " "),
        })

    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Behavioral Match Rate Over Runs")
    chart_data = {
        r["summary"].get("run_id", r["run_id"]): r["summary"].get("metrics", {}).get("behavioral_match_rate", 0)
        for r in reversed(runs)
    }
    if chart_data:
        st.bar_chart(chart_data)

# ── Run Details ───────────────────────────────────────────────────────────────

elif page == "Run Details":
    st.title("Run Details")

    if not runs:
        st.info("No evaluation runs found.")
        st.stop()

    run_ids = [r["run_id"] for r in runs]
    selected_id = st.selectbox("Select run", run_ids)
    selected_run = next(r for r in runs if r["run_id"] == selected_id)

    summary = selected_run["summary"]
    m = summary.get("metrics", {})

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Items", f"{summary.get('completed')}/{summary.get('total_items')}")
    c2.metric("Behavioral Match", _pct(m.get("behavioral_match_rate")))
    c3.metric("Clarification Correct", _pct(m.get("clarification_correct_rate")))
    c4.metric("Token Overlap", _pct(m.get("answer_token_overlap_mean")))
    c5.metric("Avg Latency", _ms(m.get("avg_response_time_ms")))
    c6.metric("Citation Presence", _pct(m.get("citation_presence_rate")))

    if summary.get("failed", 0):
        st.error(f"{summary['failed']} items failed (API errors).")
    if summary.get("error_types"):
        st.caption("Error types")
        st.dataframe(
            [{"Error Type": k, "Count": v} for k, v in summary.get("error_types", {}).items()],
            width="stretch",
            hide_index=True,
        )

    ragas = summary.get("ragas", {})
    if ragas:
        st.subheader("Ragas")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Status", ragas.get("status", "—"))
        rc2.metric("Eligible Records", ragas.get("eligible_records", 0))
        rc3.metric("Faithfulness", _pct((ragas.get("metrics") or {}).get("faithfulness")))
        if ragas.get("reason"):
            st.info(ragas.get("reason"))

    readiness_counts = summary.get("dataset_readiness", {})
    if readiness_counts:
        st.caption(f"Dataset readiness in run — ready: {readiness_counts.get('ready', 0)} · bootstrap: {readiness_counts.get('bootstrap', 0)}")

    st.subheader("By Question Type")
    by_type = summary.get("by_question_type", {})
    if by_type:
        type_rows = [
            {
                "Type": k,
                "Count": v.get("count"),
                "Behavioral Match": _pct(v.get("behavioral_match_rate")),
                "Token Overlap": _pct(v.get("token_overlap_mean")),
            }
            for k, v in by_type.items()
        ]
        st.dataframe(type_rows, width="stretch", hide_index=True)

    item_results_path = selected_run["path"] / "item_results.json"
    item_results = _load_json(item_results_path) or []

    st.subheader("Per-Item Results")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        types = ["All"] + sorted({r.get("question_type", "") for r in item_results})
        type_filter = st.selectbox("Question Type", types)
    with filter_col2:
        diffs = ["All"] + sorted({r.get("difficulty", "") for r in item_results if r.get("difficulty")})
        diff_filter = st.selectbox("Difficulty", diffs)
    with filter_col3:
        status_filter = st.selectbox("Status", ["All", "Pass", "Fail", "Error"])
    with filter_col4:
        readiness_filter = st.selectbox("Readiness", ["All", "ready", "bootstrap"])

    filtered = item_results
    if type_filter != "All":
        filtered = [r for r in filtered if r.get("question_type") == type_filter]
    if diff_filter != "All":
        filtered = [r for r in filtered if r.get("difficulty") == diff_filter]
    if readiness_filter != "All":
        filtered = [r for r in filtered if r.get("dataset_readiness") == readiness_filter]
    if status_filter == "Pass":
        filtered = [r for r in filtered if r.get("metrics", {}).get("behavioral_match") is True]
    elif status_filter == "Fail":
        filtered = [r for r in filtered if r.get("metrics", {}).get("behavioral_match") is False]
    elif status_filter == "Error":
        filtered = [r for r in filtered if r.get("status") == "error"]

    table_rows = []
    for r in filtered:
        bm = r.get("metrics", {}).get("behavioral_match")
        tok = r.get("metrics", {}).get("token_overlap_f1")
        ret = r.get("metrics", {}).get("retrieved_chunk_recall")
        table_rows.append({
            "ID": r.get("item_id"),
            "Type": r.get("question_type"),
            "Difficulty": r.get("difficulty"),
            "Readiness": r.get("dataset_readiness", "—"),
            "Status": r.get("status"),
            "Behavioral": "✓" if bm is True else "✗" if bm is False else "—",
            "Token Overlap": f"{tok:.2f}" if tok is not None else "—",
            "Chunk Recall": f"{ret:.2f}" if ret is not None else "—",
            "Attempts": r.get("attempts", 1),
            "Latency ms": r.get("response_time_ms"),
        })

    st.dataframe(table_rows, width="stretch", hide_index=True)

# ── Item Inspector ────────────────────────────────────────────────────────────

elif page == "Item Inspector":
    st.title("Item Inspector")

    if not runs:
        st.info("No evaluation runs found.")
        st.stop()

    run_ids = [r["run_id"] for r in runs]
    selected_id = st.selectbox("Select run", run_ids)
    selected_run = next(r for r in runs if r["run_id"] == selected_id)

    item_results_path = selected_run["path"] / "item_results.json"
    item_results = _load_json(item_results_path) or []

    if not item_results:
        st.info("No item results in this run.")
        st.stop()

    item_ids = [r.get("item_id", f"item-{i}") for i, r in enumerate(item_results)]
    selected_item_id = st.selectbox("Select item", item_ids)
    item = next(r for r in item_results if r.get("item_id") == selected_item_id)

    st.subheader(f"{item.get('item_id')} — {item.get('question_type')} / {item.get('difficulty')}")

    st.markdown(f"**Question:** {item.get('question', '—')}")

    col_exp, col_act = st.columns(2)
    with col_exp:
        st.markdown("#### Expected Answer")
        st.markdown(item.get("expected_answer", "—"))
    with col_act:
        st.markdown("#### Actual Answer")
        actual = item.get("actual_answer", "")
        if actual:
            st.markdown(actual)
        elif item.get("status") == "error":
            st.error(item.get("error", "API error"))
        else:
            st.warning("No answer returned.")

    st.subheader("Metrics")
    metrics = item.get("metrics", {})
    m_cols = st.columns(len(metrics) or 1)
    for col, (k, v) in zip(m_cols, metrics.items()):
        if isinstance(v, bool):
            col.metric(k.replace("_", " ").title(), "✓ Pass" if v else "✗ Fail")
        elif isinstance(v, float):
            col.metric(k.replace("_", " ").title(), f"{v:.3f}")
        else:
            col.metric(k.replace("_", " ").title(), str(v) if v is not None else "—")

    with st.expander("Full item details (JSON)"):
        st.json(item)

    retrieved_chunks = item.get("retrieved_chunks", [])
    if retrieved_chunks:
        st.subheader("Retrieved Chunks")
        st.dataframe(
            [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "section_title": chunk.get("section_title"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "score": chunk.get("score"),
                }
                for chunk in retrieved_chunks
            ],
            width="stretch",
            hide_index=True,
        )

# ── A/B Comparison ────────────────────────────────────────────────────────────

elif page == "A/B Comparison":
    st.title("A/B Comparison")

    if not ab_runs:
        st.info(
            "No A/B runs found. Run:\n\n"
            "```bash\n"
            "python evaluations/scripts/run_ab_eval.py \\\n"
            "    --variant-a-url http://localhost:8000 \\\n"
            "    --variant-b-url http://localhost:8001 \\\n"
            "    --api-key <key>\n"
            "```"
        )
        st.stop()

    ab_run_ids = [r["run_id"] for r in ab_runs]
    selected_ab_id = st.selectbox("Select A/B run", ab_run_ids)
    selected_ab = next(r for r in ab_runs if r["run_id"] == selected_ab_id)
    s = selected_ab["summary"]
    m = s.get("metrics", {})
    ma = m.get("variant_a", {})
    mb = m.get("variant_b", {})
    delta = m.get("delta_b_minus_a", {})
    wc = s.get("winner_counts", {})

    st.caption(f"Variant A: `{s.get('variant_a_url')}` · Variant B: `{s.get('variant_b_url')}`")

    def _delta_str(v: float | None, higher_is_better: bool = True) -> str:
        if v is None:
            return "—"
        sign = "+" if v > 0 else ""
        better = (v > 0) == higher_is_better
        icon = "▲" if v > 0 else ("▼" if v < 0 else "=")
        color = "green" if better else ("red" if not better and v != 0 else "grey")
        return f":{color}[{icon} {sign}{v:.4f}]"

    # ── Winner banner ──────────────────────────────────────────────────────────
    overall = s.get("overall_winner", "tie")
    if overall == "B":
        st.success(f"Variant B wins overall — A:{wc.get('A',0)}  B:{wc.get('B',0)}  Tie:{wc.get('tie',0)}")
    elif overall == "A":
        st.warning(f"Variant A wins overall — A:{wc.get('A',0)}  B:{wc.get('B',0)}  Tie:{wc.get('tie',0)}")
    else:
        st.info(f"Overall tie — A:{wc.get('A',0)}  B:{wc.get('B',0)}  Tie:{wc.get('tie',0)}")

    # ── Metric comparison cards ────────────────────────────────────────────────
    st.subheader("Global Metrics")
    metric_defs = [
        ("Behavioral Match", "behavioral_match_rate", True),
        ("Token Overlap", "answer_token_overlap_mean", True),
        ("Clarification Correct", "clarification_correct_rate", True),
        ("Citation Presence", "citation_presence_rate", True),
        ("Avg Latency", "avg_response_time_ms", False),
    ]
    cols = st.columns(len(metric_defs))
    for col, (label, key, higher_better) in zip(cols, metric_defs):
        val_a = ma.get(key)
        val_b = mb.get(key)
        d = delta.get(key)
        fmt = _ms if "time" in key else _pct
        col.markdown(f"**{label}**")
        col.markdown(f"A: `{fmt(val_a)}`  B: `{fmt(val_b)}`")
        col.markdown(_delta_str(d, higher_better))

    ragas = s.get("ragas", {})
    st.subheader("Ragas Comparison")
    st.caption(
        f"Variant A: {ragas.get('variant_a', {}).get('status', '—')} · "
        f"Variant B: {ragas.get('variant_b', {}).get('status', '—')}"
    )
    ragas_rows = []
    for key in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]:
        va = (ragas.get("variant_a", {}).get("metrics") or {}).get(key)
        vb = (ragas.get("variant_b", {}).get("metrics") or {}).get(key)
        dv = (ragas.get("delta_b_minus_a") or {}).get(key)
        ragas_rows.append({
            "Metric": key,
            "Variant A": f"{va:.3f}" if va is not None else "—",
            "Variant B": f"{vb:.3f}" if vb is not None else "—",
            "Δ B-A": f"{dv:+.3f}" if dv is not None else "—",
        })
    st.dataframe(ragas_rows, width="stretch", hide_index=True)

    # ── By question type ───────────────────────────────────────────────────────
    st.subheader("By Question Type")
    by_type = s.get("by_question_type", {})
    if by_type:
        type_rows = []
        for qt, v in sorted(by_type.items()):
            ta = v.get("variant_a", {})
            tb = v.get("variant_b", {})
            d_tok = v.get("delta_token_overlap")
            d_beh = v.get("delta_behavioral_match")
            d_lat = v.get("delta_response_time_ms")
            type_rows.append({
                "Type": qt,
                "Count": ta.get("count", "—"),
                "Token A": f"{ta.get('token_overlap_mean', 0):.3f}" if ta.get("token_overlap_mean") is not None else "—",
                "Token B": f"{tb.get('token_overlap_mean', 0):.3f}" if tb.get("token_overlap_mean") is not None else "—",
                "Δ Token": f"{'+' if d_tok and d_tok>0 else ''}{d_tok:.3f}" if d_tok is not None else "—",
                "Behav A": _pct(ta.get("behavioral_match_rate")),
                "Behav B": _pct(tb.get("behavioral_match_rate")),
                "Δ Behav": f"{'+' if d_beh and d_beh>0 else ''}{d_beh:.3f}" if d_beh is not None else "—",
                "Lat A ms": f"{ta.get('avg_response_time_ms', 0):.0f}" if ta.get("avg_response_time_ms") else "—",
                "Lat B ms": f"{tb.get('avg_response_time_ms', 0):.0f}" if tb.get("avg_response_time_ms") else "—",
            })
        st.dataframe(type_rows, width="stretch", hide_index=True)

    # ── Per-item results ───────────────────────────────────────────────────────
    ab_items_path = selected_ab["path"] / "ab_item_results.json"
    ab_items = _load_json(ab_items_path) or []

    if ab_items:
        st.subheader("Per-Item Results")

        filter_type = st.selectbox(
            "Filter by type",
            ["All"] + sorted({r.get("question_type", "") for r in ab_items}),
        )
        filter_winner = st.selectbox("Filter by winner", ["All", "A", "B", "tie"])

        filtered = ab_items
        if filter_type != "All":
            filtered = [r for r in filtered if r.get("question_type") == filter_type]
        if filter_winner != "All":
            filtered = [r for r in filtered if r.get("winner") == filter_winner]

        item_rows = []
        for r in filtered:
            ra = r.get("variant_a", {})
            rb = r.get("variant_b", {})
            d = r.get("delta_b_minus_a", {})
            tok_a = (ra.get("metrics") or {}).get("token_overlap_f1")
            tok_b = (rb.get("metrics") or {}).get("token_overlap_f1")
            item_rows.append({
                "ID": r.get("item_id"),
                "Type": r.get("question_type"),
                "Difficulty": r.get("difficulty"),
                "Winner": r.get("winner", "—"),
                "Readiness": r.get("dataset_readiness", "—"),
                "Tok A": f"{tok_a:.3f}" if tok_a is not None else "—",
                "Tok B": f"{tok_b:.3f}" if tok_b is not None else "—",
                "Δ Tok": f"{'+' if d.get('token_overlap_f1', 0) > 0 else ''}{d['token_overlap_f1']:.3f}" if d.get("token_overlap_f1") is not None else "—",
                "Lat A": f"{ra.get('response_time_ms', 0):.0f} ms",
                "Lat B": f"{rb.get('response_time_ms', 0):.0f} ms",
            })
        st.dataframe(item_rows, width="stretch", hide_index=True)

        # ── Item drill-down ────────────────────────────────────────────────────
        st.subheader("Item Side-by-Side")
        all_item_ids = [r.get("item_id") for r in ab_items]
        selected_ab_item_id = st.selectbox("Select item", all_item_ids)
        ab_item = next(r for r in ab_items if r.get("item_id") == selected_ab_item_id)

        st.markdown(f"**Question:** {ab_item.get('question', '—')}")
        st.markdown(f"**Expected:** {ab_item.get('expected_answer', '—')[:300]}{'…' if len(ab_item.get('expected_answer',''))>300 else ''}")

        col_a, col_b = st.columns(2)
        with col_a:
            va = ab_item.get("variant_a", {})
            st.markdown("#### Variant A")
            ans_a = va.get("answer", "")
            st.markdown(ans_a if ans_a else "*No answer*")
            st.caption(f"Latency: {va.get('response_time_ms', 0):.0f} ms · Citations: {va.get('api_citations_count', 0)}")
            for k, v in (va.get("metrics") or {}).items():
                if isinstance(v, float):
                    st.metric(k, f"{v:.3f}")
        with col_b:
            vb = ab_item.get("variant_b", {})
            st.markdown("#### Variant B")
            ans_b = vb.get("answer", "")
            st.markdown(ans_b if ans_b else "*No answer*")
            st.caption(f"Latency: {vb.get('response_time_ms', 0):.0f} ms · Citations: {vb.get('api_citations_count', 0)}")
            for k, v in (vb.get("metrics") or {}).items():
                if isinstance(v, float):
                    st.metric(k, f"{v:.3f}", delta=ab_item.get("delta_b_minus_a", {}).get(k))

        with st.expander("Full A/B item JSON"):
            st.json(ab_item)
