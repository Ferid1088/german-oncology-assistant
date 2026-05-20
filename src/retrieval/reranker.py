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
