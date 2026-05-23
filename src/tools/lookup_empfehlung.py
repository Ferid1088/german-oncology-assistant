"""Empfehlung lookup tool: fetches a specific clinical recommendation by its dotted ID.

Performs a direct Milvus exact-match query for ``recommendation_id`` within a
named guideline, returning the verbatim text with grade and evidence level.
Used when the user references a specific recommendation number (e.g. "4.2.1").
"""

import os
from pymilvus import MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI") or "./milvus.db"
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
    # Milvus filter expression: exact match on all three fields to avoid
    # returning prose chunks that share the same section number prefix.
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
