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

    # Retrieval
    retrieved_chunks: list[dict]
    confidence: float                    # 0.0–1.0 from reranker scores

    # Generation
    answer_professional: str
    answer_plain: str
    citations: list[dict]
    disclaimer: str

    # Guardrails
    input_blocked: bool
    input_block_reason: str
    output_blocked: bool

    # Tool calls (for display)
    tool_calls_log: list[dict]

    # Conversation memory
    messages: Annotated[list, add_messages]
