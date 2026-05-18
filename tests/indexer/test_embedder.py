import pytest
from unittest.mock import MagicMock
from src.indexer.embedder import embed_texts


def test_embed_texts_returns_correct_shape():
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 3072, index=0), MagicMock(embedding=[0.2] * 3072, index=1)]
    )
    result = embed_texts(["text one", "text two"], client=client)
    assert len(result) == 2
    assert len(result[0]) == 3072


def test_embed_texts_preserves_order():
    client = MagicMock()
    # API returns items in REVERSED order — index field must be used to restore correct order
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.2] * 3072, index=1), MagicMock(embedding=[0.1] * 3072, index=0)]
    )
    result = embed_texts(["text one", "text two"], client=client)
    # result[0] must be index=0's embedding (0.1), not index=1's (0.2)
    assert result[0][0] == pytest.approx(0.1)
    assert result[1][0] == pytest.approx(0.2)
