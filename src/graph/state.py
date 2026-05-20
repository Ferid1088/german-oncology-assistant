from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    # Input
    user_query: str
    session_id: str

    # Preprocessing
    rewritten_query: str
    metadata_filters: dict[str, str]    # guideline_id, grade, chunk_type
    intent: str                          # factual | recommendation | comparison | external
    query_decomposition: list[str]
    requires_clarification: bool
    missing_clinical_dimensions: list[str]
    clarification_rationale: str | None
    expected_clarification: str | None
    user_role: str
    allowed_sources: list[str]

    # Retrieval
    retrieved_chunks: list[dict]
    confidence: float                    # 0.0–1.0 from reranker scores
    escalation_reason: str

    # Generation
    answer_professional: str
    answer_plain: str
    citations: list[dict]
    disclaimer: str

    # Guardrails
    input_blocked: bool
    input_block_reason: str
    output_blocked: bool
    redacted_query: str

    # Tool calls (for display)
    tool_calls_log: list[dict]

    # Turn understanding / conversation reuse
    turn_intents: list[str]
    followup_routing: str                  # memory | retrieve
    prior_answer_professional: str
    prior_answer_plain: str
    prior_citations: list[dict]
    prior_retrieved_chunks: list[dict]

    # Conversation memory
    messages: Annotated[list, add_messages]
