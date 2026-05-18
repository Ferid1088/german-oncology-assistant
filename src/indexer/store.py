import os
from pymilvus import MilvusClient, DataType

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")
DIM = 3072


class MilvusStore:
    def __init__(self, collection_name: str = COLLECTION, client: MilvusClient | None = None):
        self.collection_name = collection_name
        self.client = client or MilvusClient(uri=MILVUS_URI)

    def ensure_collection(self) -> None:
        """Create collection with dense HNSW index if it does not exist."""
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("guideline_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
        schema.add_field("recommendation_grade", DataType.VARCHAR, max_length=8)
        schema.add_field("is_leaf", DataType.BOOL)

        index_params = self.client.prepare_index_params()
        index_params.add_index("dense_vector", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, records: list[dict]) -> None:
        """Insert or update records. Each dict must have chunk_id and dense_vector."""
        self.client.insert(collection_name=self.collection_name, data=records)

    def drop(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
