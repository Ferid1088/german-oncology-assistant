from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_eval_dataset.py"


def test_generate_eval_dataset_script(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout.strip())
    assert summary["status"] == "ok"

    dataset_path = tmp_path / "evaluation-dataset.json"
    guardrail_path = tmp_path / "guardrail-dataset.json"
    smoke_path = tmp_path / "smoke-dataset.json"
    structure_path = tmp_path / "evaluation-dataset.structure.json"

    assert dataset_path.exists()
    assert guardrail_path.exists()
    assert smoke_path.exists()
    assert structure_path.exists()

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    guardrail_dataset = json.loads(guardrail_path.read_text(encoding="utf-8"))
    smoke_dataset = json.loads(smoke_path.read_text(encoding="utf-8"))
    structure = json.loads(structure_path.read_text(encoding="utf-8"))

    assert len(dataset) == 30
    assert structure["target_dataset_size"] == 30
    assert structure["recommended_file_layout"]["main_eval_dataset"].endswith("evaluation-dataset.json")

    counts = Counter(item["question_type"] for item in dataset)
    assert counts == Counter(
        {
            "recommendation": 10,
            "factual": 6,
            "evidence": 5,
            "comparison": 4,
            "drug_lookup": 3,
            "external": 2,
        }
    )

    for item in dataset:
        assert item["dataset_split"] == "eval"
        assert "question" in item and item["question"]
        assert "expected_answer" in item
        assert isinstance(item["guideline_scope"], list)
        assert isinstance(item["expected_tools"], list)
        assert isinstance(item["requires_clarification"], bool)
        assert isinstance(item["missing_clinical_dimensions"], list)
        assert isinstance(item["required_citations"], list)
        assert isinstance(item["expected_answer_sections"], list)
        assert isinstance(item["retrieval_challenge_types"], list)

    clarification_items = [item for item in dataset if item["requires_clarification"]]
    assert clarification_items
    assert all(item["expected_behavior"] == "ask_clarification" for item in clarification_items)
    assert all(item["expected_clarification"] for item in clarification_items)

    assert len(guardrail_dataset) >= 4
    assert all(item["dataset_split"] == "guardrail" for item in guardrail_dataset)
    assert all(item["question_type"] == "guardrail" for item in guardrail_dataset)

    assert len(smoke_dataset) == 4
    assert all(item["dataset_split"] == "smoke" for item in smoke_dataset)

    recommendation_items = [
        item for item in dataset
        if item["question_type"] == "recommendation" and not item["requires_clarification"]
    ]
    assert recommendation_items
    assert all(item["expected_recommendation_metadata"] is not None for item in recommendation_items)

    assert "required_citations" in structure["field_groups_for_next_evaluation_steps"]["retrieval_and_citation"]
    assert "recommendation_fidelity" in structure["field_groups_for_next_evaluation_steps"]
    assert "retrieval_debugging" in structure["field_groups_for_next_evaluation_steps"]