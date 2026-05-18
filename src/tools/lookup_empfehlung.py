import os
from pymilvus import MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")


def lookup_empfehlung_tool(
    guideline_id: str,
    recommendation_id: str,
    client: MilvusClient | None = None,
) -> dict:
    """
    Fetch a specific Empfehlung X.Y verbatim by its recommendation_id.
    Returns the full text with grade, evidence level, and source metadata.
    """
    c = client or MilvusClient(uri=MILVUS_URI)
    expr = (
        f'guideline_id == "{guideline_id}" and '
        f'recommendation_id == "{recommendation_id}" and '
        f'chunk_type == "recommendation"'
    )
    rows = c.query(
        collection_name=COLLECTION,
        filter=expr,
        output_fields=["chunk_id", "text", "recommendation_grade", "evidence_level", "section_title", "guideline_id"],
        limit=1,
    )
    if not rows:
        return {
            "found": False,
            "recommendation_id": recommendation_id,
            "guideline_id": guideline_id,
            "message": f"Empfehlung {recommendation_id} nicht gefunden in {guideline_id}.",
        }
    row = rows[0]
    return {
        "found": True,
        "recommendation_id": recommendation_id,
        "guideline_id": row["guideline_id"],
        "text": row["text"],
        "recommendation_grade": row["recommendation_grade"],
        "evidence_level": row["evidence_level"],
        "section_title": row["section_title"],
    }
