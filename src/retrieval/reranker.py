import logging
from src.retrieval.search import RetrievedChunk

log = logging.getLogger(__name__)

try:
    from FlagEmbedding import FlagReranker as _FlagReranker
    _RERANKER_AVAILABLE = True
except Exception:
    _RERANKER_AVAILABLE = False

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = _FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    if not _RERANKER_AVAILABLE:
        return chunks[:top_k]
    try:
        reranker = _get_reranker()
        pairs = [[query, c.text] for c in chunks]
        scores = reranker.compute_score(pairs, normalize=True)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(**{**vars(c), "score": float(s)})
            for s, c in ranked[:top_k]
        ]
    except Exception as e:
        log.warning("Reranker failed (%s), returning top-%d by dense score", e, top_k)
        return chunks[:top_k]
