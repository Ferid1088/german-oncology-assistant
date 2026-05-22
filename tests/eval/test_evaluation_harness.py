from __future__ import annotations

from evaluations.scripts.check_dataset import run as run_dataset_check
from evaluations.scripts.run_eval import evaluate_item
from evaluations.utils import bootstrap_reasons, is_live_eval_ready


def test_bootstrap_readiness_helpers_detect_seed_ids_and_missing_answers():
    item = {
        "expected_behavior": "answer",
        "expected_answer": None,
        "source_chunk_ids": ["seed:mamma:example"],
        "gold_chunk_ids": [],
        "required_citations": [],
    }

    reasons = bootstrap_reasons(item)

    assert "source_chunk_ids" in reasons
    assert "missing_expected_answer" in reasons
    assert is_live_eval_ready(item) is False


def test_dataset_check_reports_readiness_warnings_for_bootstrap_dataset():
    result = run_dataset_check(
        dataset_path="data/eval/evaluation-dataset.json",
        schema_path="docs/evaluation-dataset.schema.json",
        split="eval",
    )

    assert result["schema_valid"] is True
    assert result["blocking_ok"] is True
    assert result["readiness_ok"] is False
    assert result["items_ready_for_eval"] < result["total_items"]
    assert result["selected_split"] == "eval"
    assert result["bootstrap_reason_counts"]


def test_evaluate_item_handles_clarification_even_when_flag_missing_in_payload():
    item = {
        "id": "eval-x",
        "question": "Welche Chemotherapie wird empfohlen?",
        "question_type": "recommendation",
        "difficulty": "medium",
        "expected_behavior": "ask_clarification",
        "requires_clarification": True,
        "expected_answer": None,
        "expected_answer_sections": ["clarification"],
        "required_citations": [],
        "expected_tools": [],
        "expected_recommendation_metadata": None,
        "claim_verdict": None,
        "source_chunk_ids": [],
        "gold_chunk_ids": [],
    }
    resp = {
        "answer_professional": "Ich brauche vor der Leitlinienrecherche noch eine Präzisierung Ihrer Frage. Bitte präzisieren Sie die klinische Situation.",
        "answer_plain": "",
        "requires_clarification": False,
        "citations": [],
        "tool_calls": [],
        "blocked": False,
        "token_usage": {},
        "safety_warning": None,
    }

    result = evaluate_item(item, resp, elapsed_ms=1234.5)

    assert result["status"] == "completed"
    assert result["metrics"]["behavioral_match"] is True
    assert result["metrics"]["clarification_correct"] is True
    assert result["metrics"]["expected_sections_coverage"] == 1.0


def test_evaluate_item_computes_retrieval_grounding_metrics_from_retrieved_chunks():
    item = {
        "id": "eval-ret",
        "question": "Was empfiehlt die Leitlinie?",
        "question_type": "recommendation",
        "difficulty": "easy",
        "expected_behavior": "answer",
        "requires_clarification": False,
        "expected_answer": "Referenzantwort",
        "expected_answer_sections": ["recommendation"],
        "required_citations": [{"chunk_id": "c1", "citation_importance": "must"}],
        "expected_tools": [],
        "expected_recommendation_metadata": None,
        "claim_verdict": None,
        "source_chunk_ids": ["c1"],
        "gold_chunk_ids": ["c1"],
        "gold_sections": ["Therapie"],
        "gold_pages": [12],
    }
    resp = {
        "answer_professional": "Referenzantwort",
        "answer_plain": "Einfach erklärt",
        "requires_clarification": False,
        "citations": [{"chunk_id": "c1"}],
        "retrieved_chunks": [{"chunk_id": "c1", "section_title": "Therapie", "page_start": 12, "page_end": 12}],
        "tool_calls": [],
        "blocked": False,
        "token_usage": {},
        "safety_warning": None,
    }

    result = evaluate_item(item, resp, elapsed_ms=250.0)

    assert result["metrics"]["retrieved_chunk_recall"] == 1.0
    assert result["metrics"]["retrieved_chunk_precision"] == 1.0
    assert result["metrics"]["retrieved_section_recall"] == 1.0
    assert result["metrics"]["retrieved_page_recall"] == 1.0
    assert result["metrics"]["citation_chunk_recall"] == 1.0