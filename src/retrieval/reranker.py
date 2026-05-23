"""CrossEncoder reranker: re-scores retrieved chunks using a bi-encoder-free model.

Uses ``BAAI/bge-reranker-v2-m3`` from sentence-transformers, which is a
cross-encoder — it jointly encodes the query and each candidate passage and
produces a more accurate relevance score than embedding cosine similarity alone.

The reranker is loaded lazily on the first call and cached as a module-level
singleton.  If the model fails to load (e.g. sentence-transformers not installed,
CUDA OOM), a permanent ``_reranker_unavailable`` flag is set and all subsequent
calls fall back to returning results in their original retrieval order.
"""

import logging
import os

from src.retrieval.search import RetrievedChunk

log = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _RERANKER_AVAILABLE = True
except Exception:
    _RERANKER_AVAILABLE = False

_reranker = None
_reranker_unavailable = False
_logged_unavailable = False

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "12"))


def _get_reranker():
    """Return the cached CrossEncoder instance, loading it on first call.

    Raises:
        RuntimeError: If a previous load attempt failed (permanent failure).
        Exception: Propagates the original load error and sets the unavailable flag.
    """
    global _reranker, _reranker_unavailable
    if _reranker_unavailable:
        raise RuntimeError("Reranker disabled after previous load failure")
    if _reranker is None:
        try:
            _reranker = _CrossEncoder(RERANKER_MODEL)
        except Exception:
            _reranker_unavailable = True
            raise
    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Re-score and reorder *chunks* using the CrossEncoder reranker.

    Caps the candidate pool at ``RERANKER_CANDIDATES`` (default 12) to keep
    latency predictable — the pool is always ≥ top_k to avoid re-ranking fewer
    candidates than requested.  Falls back gracefully if the reranker is
    unavailable, returning the original retrieval-order top-k.

    Args:
        query: User query string (used as the left side of each (query, passage) pair).
        chunks: Retrieved chunks in retrieval-score order.
        top_k: Number of top-ranked chunks to return.

    Returns:
        Up to *top_k* chunks with their ``score`` field updated to the
        CrossEncoder relevance score.  Returns retrieval-order top-k on failure.
    """
    global _logged_unavailable
    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks[:top_k]
    if not _RERANKER_AVAILABLE:
        if not _logged_unavailable:
            log.warning("sentence-transformers CrossEncoder unavailable, reranker is disabled")
            _logged_unavailable = True
        return chunks[:top_k]

    candidate_count = max(top_k, min(len(chunks), RERANKER_CANDIDATES))
    candidate_pool = chunks[:candidate_count]

    try:
        reranker = _get_reranker()
        pairs = [(query, c.text) for c in candidate_pool]
        scores = reranker.predict(
            pairs,
            show_progress_bar=False,
            batch_size=RERANKER_BATCH_SIZE,
        )
        ranked = sorted(zip(scores, candidate_pool), key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(**{**vars(c), "score": float(s)})
            for s, c in ranked[:top_k]
        ]
    except Exception as e:
        _logged_unavailable = True
        log.warning("Reranker failed (%s), returning top-%d by retrieval score", e, top_k)
        return chunks[:top_k]
