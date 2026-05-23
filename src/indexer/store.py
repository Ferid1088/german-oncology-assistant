"""Milvus vector store wrapper for the oncology guideline collection.

Manages collection lifecycle (create, upsert, drop) for the HNSW dense index.
The collection schema uses:
- ``chunk_id`` (VARCHAR, primary key) — UUID assigned at index time.
- ``dense_vector`` (FLOAT_VECTOR, 3072 dim) — embedding from text-embedding-3-large.
- ``is_leaf``, ``guideline_id``, ``chunk_type``, ``recommendation_grade`` as
  filterable scalar fields.
- Dynamic fields enabled so enricher metadata is stored without schema changes.

HNSW parameters: M=16, efConstruction=200 — standard values balancing build
time against recall at this corpus size (~tens of thousands of chunks).
"""

import os
from pymilvus import MilvusClient, DataType

MILVUS_URI = os.getenv("MILVUS_URI") or "./milvus.db"
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")
DIM = 3072


class MilvusStore:
    """High-level interface for upserting and managing the guideline chunk collection.

    Attributes:
        collection_name: Milvus collection to operate on (default: ``oncology_guidelines``).
        client: Underlying MilvusClient instance.
    """

    def __init__(self, collection_name: str = COLLECTION, client: MilvusClient | None = None):
        """Initialise the store, connecting to Milvus Lite or a remote instance.

        Args:
            collection_name: Target Milvus collection name.
            client: Optional pre-built client (used in tests to inject a mock).
        """
        self.collection_name = collection_name
        self.client = client or MilvusClient(uri=MILVUS_URI)

    def ensure_collection(self) -> None:
        """Create collection with dense HNSW index if it does not exist."""
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=16384)
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
        self.client.load_collection(self.collection_name)

    def upsert(self, records: list[dict]) -> None:
        """Insert or update records. Each dict must have chunk_id and dense_vector."""
        self.client.upsert(collection_name=self.collection_name, data=records)

    def drop(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
