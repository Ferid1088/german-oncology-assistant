"""Shared LangGraph state schema for the oncology RAG pipeline.

Every node in the graph reads from and writes to a single ``RAGState`` instance.
Fields are grouped by the pipeline stage that owns them.  Nodes must only write
fields they are responsible for; all other fields are carried through unchanged.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """Central state container passed between all LangGraph nodes.

    Field groups:
    - **Input**: raw user-supplied values for the current turn.
    - **Preprocessing**: normalised query and routing metadata produced by the
      rewrite and turn-router nodes.
    - **Retrieval**: chunks returned by the agent tool-calling loop.
    - **Generation**: final answer text and citation metadata.
    - **Guardrails**: flags and reasons set by input/output safety nodes.
    - **Telemetry**: per-call token usage, cost, and full pipeline trace.
    - **Turn routing**: intent classification and conversation-reuse flags.
    - **Conversation memory**: LangChain message history with merge semantics.
    """

    # ------------------------------------------------------------------ Input
    user_query: str          # Raw query exactly as submitted by the user.
    session_id: str          # Opaque identifier that groups turns into a conversation.

    # -------------------------------------------------------------- Preprocessing
    rewritten_query: str                  # Normalised, clinically expanded query.
    metadata_filters: dict[str, str]      # Extracted Milvus filters: guideline_id, grade, chunk_type.
    intent: str                           # Query intent: factual | recommendation | comparison | external.
    query_decomposition: list[str]        # Sub-queries when the query spans multiple topics.
    requires_clarification: bool          # True when the rewriter needs more clinical detail.
    missing_clinical_dimensions: list[str]  # E.g. ["disease_stage", "line_of_therapy"].
    clarification_rationale: str | None   # Why clarification is needed (shown to the user).
    expected_clarification: str | None    # The specific follow-up question asked of the user.
    user_role: str                        # RBAC role: user | professional | admin.
    allowed_sources: list[str]            # Permitted data sources: guidelines | web | pubmed.

    # --------------------------------------------------------------- Retrieval
    retrieved_chunks: list[dict]   # Ranked, deduplicated chunks from the agent tool loop.
    confidence: float              # Mean reranker score of top-3 chunks (0.0–1.0).
    escalation_reason: str         # Why escalation was triggered: low_score | low_result_count | "".

    # --------------------------------------------------------------- Generation
    answer_professional: str   # Full clinical answer for healthcare professionals.
    answer_plain: str          # Plain-language summary for patients or lay readers.
    citations: list[dict]      # Citation metadata for chunks referenced in the answer.
    disclaimer: str            # Standard guideline-only disclaimer appended to every answer.

    # --------------------------------------------------------------- Guardrails
    input_blocked: bool            # True when the input guardrail rejected the request.
    input_block_reason: str        # German-language explanation returned to the user.
    output_blocked: bool           # True when the output guardrail suppressed the answer.
    redacted_query: str            # Query with PII replaced by [REDACTED].
    safety_warning: str | None     # Short English safety label (shown in UI).
    safety_explanation: str | None # Detailed explanation of why the output was limited.
    safety_title: str | None       # UI heading for the safety panel.

    # --------------------------------------------------------------- Telemetry
    tool_calls_log: list[dict]           # One entry per tool invocation (tool, args, summary, status).
    rag_trace: list[dict]                # Ordered pipeline steps: name, status, summary, duration_ms.
    token_usage: dict                    # Aggregated token counts and USD cost across all LLM calls.
    external_search_snippets: list[dict] # Web/DuckDuckGo result snippets appended after main answer.

    # ----------------------------------------------- Turn routing / conversation reuse
    turn_intents: list[str]  # Classified intents for this turn, e.g. ["clarify", "new_query"].
    followup_routing: str    # Routing decision: "memory" (reuse prior answer) | "retrieve" (full pipeline).
    prior_answer_professional: str        # Previous turn's professional answer (for memory reuse).
    prior_answer_plain: str               # Previous turn's plain answer.
    prior_citations: list[dict]           # Previous turn's citations.
    prior_retrieved_chunks: list[dict]    # Previous turn's retrieved chunks.
    prior_rewritten_query: str            # Previous turn's rewritten query (used for duplicate detection).
    prior_rag_trace: list[dict]           # Previous turn's pipeline trace (used to detect prior disclaimer).
    prior_external_search_snippets: list[dict]  # Previous turn's web snippets.

    # -------------------------------------------------------- Conversation memory
    # add_messages merges incoming message lists rather than overwriting,
    # preserving the full multi-turn conversation history across graph invocations.
    messages: Annotated[list, add_messages]
