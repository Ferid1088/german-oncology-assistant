"""Post-processing utilities for retrieved chunk lists.

Applied after reranking to deduplicate and limit the final result set before
chunks are passed to the answer generation node.
"""

from math import inf


def top_unique_result_dicts(results: list[dict], top_k: int) -> list[dict]:
    """Keep the best-scoring hit per chunk_id and return a score-sorted top-k list."""
    best_by_id: dict[str, dict] = {}
    without_id: list[dict] = []

    for result in results:
        chunk_id = result.get("chunk_id")
        if not chunk_id:
            without_id.append(result)
            continue

        existing = best_by_id.get(chunk_id)
        score = result.get("score", -inf)
        existing_score = existing.get("score", -inf) if existing else -inf
        if existing is None or score > existing_score:
            best_by_id[chunk_id] = result

    ranked = sorted(best_by_id.values(), key=lambda item: item.get("score", -inf), reverse=True)
    passthrough = sorted(without_id, key=lambda item: item.get("score", -inf), reverse=True)
    return (ranked + passthrough)[:top_k]