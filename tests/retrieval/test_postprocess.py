from src.retrieval.postprocess import top_unique_result_dicts


def test_top_unique_result_dicts_keeps_best_score_per_chunk_and_sorts():
    results = [
        {"chunk_id": "a", "score": 0.20, "text": "older a"},
        {"chunk_id": "b", "score": 0.95, "text": "best b"},
        {"chunk_id": "a", "score": 0.80, "text": "better a"},
        {"chunk_id": "c", "score": 0.70, "text": "good c"},
    ]

    ranked = top_unique_result_dicts(results, top_k=3)

    assert [item["chunk_id"] for item in ranked] == ["b", "a", "c"]
    assert ranked[1]["text"] == "better a"