"""Text embedding module: batched vector generation via OpenRouter.

Wraps the OpenAI embeddings API (accessed through OpenRouter) and splits large
input lists into fixed-size batches to stay within API request size limits.
Results are sorted by their response index to guarantee ordering consistency.
"""

import os
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
# Max texts per API call; balances request size against latency.
EMBED_BATCH_SIZE = 64


def _client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """Embed a list of texts in batches. Returns list of 3072-dim vectors."""
    c = client or _client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = c.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        sorted_data = sorted(resp.data, key=lambda item: item.index)
        all_embeddings.extend([item.embedding for item in sorted_data])
    return all_embeddings
