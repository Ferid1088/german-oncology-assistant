from FlagEmbedding import FlagReranker
from src.retrieval.search import RetrievedChunk

_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    reranker = _get_reranker()
    pairs = [[query, c.text] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [
        RetrievedChunk(**{**vars(c), "score": float(s)})
        for s, c in ranked[:top_k]
    ]
