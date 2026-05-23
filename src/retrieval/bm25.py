"""
BM25 sparse retrieval over the Milvus chunk corpus.

Index lifecycle:
  - Built by scripts/run_indexer.py after every Milvus upsert and saved to BM25_INDEX_PATH.
  - Loaded once at API startup via _get_index(); never rebuilt at query time.

The index stores (chunk_ids, tokenised_corpus) so a query can be scored and
chunk_ids returned; the caller then fetches full metadata from the RetrievedChunk
objects already held in the dense result, or from Milvus if needed.
"""
import logging
import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl"))

# Simple whitespace + punctuation tokeniser that keeps German compound words intact.
_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    """Split *text* into lowercase tokens, preserving German compound words.

    Uses a punctuation-boundary tokeniser rather than whitespace-only so that
    hyphenated compounds (``"nicht-kleinzelligem"``) are split correctly while
    umlauts and ß are retained within tokens.

    Args:
        text: Raw chunk or query text.

    Returns:
        List of lowercase token strings with empty tokens removed.
    """
    return [t.lower() for t in _TOKEN_RE.split(text) if t]


# ---------------------------------------------------------------------------
# Index build (called from run_indexer.py)
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[dict], path: Path = BM25_INDEX_PATH) -> None:
    """
    Build a BM25Okapi index from a list of chunk dicts (must have 'chunk_id' and 'text')
    and persist it to *path*.
    """
    chunk_ids = [c["chunk_id"] for c in chunks]
    corpus = [_tokenise(c.get("text", "")) for c in chunks]
    index = BM25Okapi(corpus)
    with open(path, "wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "index": index}, f)
    log.info("BM25 index saved to %s (%d chunks)", path, len(chunk_ids))


# ---------------------------------------------------------------------------
# Index load (singleton, loaded once at search time)
# ---------------------------------------------------------------------------

_cache: dict = {}


def _get_index() -> tuple[list[str], BM25Okapi] | None:
    if "data" not in _cache:
        if not BM25_INDEX_PATH.exists():
            log.warning("BM25 index not found at %s — run the indexer first.", BM25_INDEX_PATH)
            return None
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _cache["data"] = data
        log.info("BM25 index loaded from %s (%d chunks)", BM25_INDEX_PATH, len(data["chunk_ids"]))
    d = _cache["data"]
    return d["chunk_ids"], d["index"]


def reload_bm25_index() -> None:
    """Force a reload on the next query (call after re-indexing)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    top_k: int = 20,
    guideline_id: str | None = None,
    chunk_id_whitelist: set[str] | None = None,
) -> list:
    """
    Return up to *top_k* RetrievedChunk stubs ranked by BM25 score.

    Only chunk_id and score are meaningful in the returned objects; all other
    fields are empty. The caller is expected to merge these stubs with dense
    results via rrf_fuse(), which carries the full metadata from the dense side.

    *chunk_id_whitelist*: if provided, only chunks whose IDs are in this set
    are considered (used to apply guideline_id filter without loading all metadata).
    """
    result = _get_index()
    if result is None:
        return []
    chunk_ids, index = result

    tokens = _tokenise(query)
    if not tokens:
        return []

    scores = index.get_scores(tokens)

    ranked = sorted(
        ((score, cid) for score, cid in zip(scores, chunk_ids)
         if chunk_id_whitelist is None or cid in chunk_id_whitelist),
        reverse=True,
    )[:top_k]

    from src.retrieval.search import RetrievedChunk  # deferred to avoid circular import
    return [
        RetrievedChunk(
            chunk_id=cid,
            text="",
            score=float(score),
            guideline_id=guideline_id or "",
            section_title="",
            section_path=[],
            page_start=None,
            page_end=None,
            chunk_type="",
            recommendation_grade="",
        )
        for score, cid in ranked
        if score > 0
    ]
