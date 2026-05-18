from src.indexer.store import MilvusStore


def test_milvus_store_upsert_calls_insert(mocker):
    mock_client = mocker.MagicMock()
    store = MilvusStore(collection_name="test_col", client=mock_client)
    store.upsert([{
        "chunk_id": "abc",
        "text": "test",
        "dense_vector": [0.1] * 3072,
        "guideline_id": "mamma",
        "chunk_type": "section",
        "section_path": ["1"],
        "recommendation_grade": "",
        "is_leaf": True,
    }])
    mock_client.insert.assert_called_once()
