import json
import os
import logging
from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from langchain_core.messages import AIMessage, HumanMessage
from src.api.auth import verify_api_key
from src.api.conversation_store import get_conversation_store
from src.api.observability import build_technical_details, get_trace_id, log_event
from src.api.rate_limit import enforce_rate_limit, route_group_for_path
from src.graph.graph import build_graph
from src.graph.checkpointing import build_checkpointer
from src.graph.state import RAGState

router = APIRouter()
_graph = None
log = logging.getLogger(__name__)
DEFAULT_USER_ROLE = os.getenv("DEFAULT_USER_ROLE", "user")
DEFAULT_ALLOWED_SOURCES = ["guidelines", "web"]


def _combine_answer_parts(answer_professional: str, answer_plain: str) -> str:
    if not answer_plain:
        return (answer_professional or "").strip()

    parts = []
    if answer_professional:
        parts.append(f"Ola antwortet fachlich:\n{answer_professional}")
    if answer_plain:
        parts.append(f"Ola erklärt es einfach:\n{answer_plain}")
    return "\n\n".join(parts).strip()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=build_checkpointer())
    return _graph


def _supports_checkpointing(graph) -> bool:
    return getattr(graph, "checkpointer", None) is not None


def _load_session_memory(session_id: str) -> dict:
    return get_conversation_store().load_session_memory(session_id)


def _save_session_memory(session_id: str, final_state: dict, user_query: str) -> None:
    get_conversation_store().append_turn(
        conversation_id=session_id,
        user_query=user_query,
        final_state=final_state,
        combined_answer=_combine_answer_parts(
            final_state.get("answer_professional", ""),
            final_state.get("answer_plain", ""),
        ),
    )


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=1500)
    session_id: str = Field(default="default", min_length=1, max_length=120)
    guideline_id: str = ""
    grade: str = ""

    @field_validator("guideline_id")
    @classmethod
    def validate_guideline_id(cls, value: str) -> str:
        allowed = {"", "mamma", "krk", "lunge", "prosta"}
        if value not in allowed:
            raise ValueError("guideline_id must be one of: mamma, krk, lunge, prosta")
        return value

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, value: str) -> str:
        allowed = {"", "A", "B", "0"}
        if value not in allowed:
            raise ValueError("grade must be one of: A, B, 0")
        return value


@router.post("/chat")
def chat(request: ChatRequest, raw_request: Request):
    # Sync endpoint — FastAPI runs this in a thread pool, so blocking LLM calls are safe.
    trace_id = get_trace_id(raw_request)
    try:
        api_key = verify_api_key(raw_request)
        enforce_rate_limit(raw_request, api_key=api_key, route_group=route_group_for_path(raw_request.url.path))
        log_event(
            log,
            "chat_request_started",
            trace_id=trace_id,
            session_id=request.session_id,
            guideline_id=request.guideline_id or None,
            grade=request.grade or None,
        )
        graph = get_graph()
        use_checkpoint = _supports_checkpointing(graph)
        session_memory = _load_session_memory(request.session_id)
        input_messages = [HumanMessage(content=request.query)] if use_checkpoint else list(session_memory.get("messages", []))
        if not use_checkpoint:
            input_messages.append(HumanMessage(content=request.query))
        initial_state = RAGState(
            user_query=request.query,
            session_id=request.session_id,
            rewritten_query="",
            metadata_filters={k: v for k, v in {
                "guideline_id": request.guideline_id,
                "grade": request.grade,
            }.items() if v},
            intent="",
            query_decomposition=[],
            requires_clarification=False,
            missing_clinical_dimensions=[],
            clarification_rationale=None,
            expected_clarification=None,
            user_role=DEFAULT_USER_ROLE,
            allowed_sources=DEFAULT_ALLOWED_SOURCES,
            retrieved_chunks=[],
            confidence=0.0,
            escalation_reason="",
            answer_professional="",
            answer_plain="",
            citations=[],
            disclaimer="",
            input_blocked=False,
            input_block_reason="",
            output_blocked=False,
            redacted_query=request.query,
            tool_calls_log=[],
            rag_trace=[],
            token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "currency": "USD", "calls": []},
            external_search_snippets=[],
            turn_intents=[],
            followup_routing="retrieve",
            prior_answer_professional=session_memory.get("prior_answer_professional", ""),
            prior_answer_plain=session_memory.get("prior_answer_plain", ""),
            prior_citations=session_memory.get("prior_citations", []),
            prior_retrieved_chunks=session_memory.get("prior_retrieved_chunks", []),
            prior_rewritten_query=session_memory.get("prior_rewritten_query", ""),
            prior_rag_trace=session_memory.get("prior_rag_trace", []),
            prior_external_search_snippets=session_memory.get("prior_external_search_snippets", []),
            messages=input_messages,
        )

        if use_checkpoint:
            config = {"configurable": {"thread_id": request.session_id}}
            final_state = graph.invoke(initial_state, config=config)
            graph.update_state(
                config,
                {
                    "messages": [AIMessage(content=_combine_answer_parts(
                        final_state.get("answer_professional", ""),
                        final_state.get("answer_plain", ""),
                    ))],
                    "prior_answer_professional": final_state["answer_professional"],
                    "prior_answer_plain": final_state["answer_plain"],
                    "prior_citations": final_state["citations"],
                    "prior_retrieved_chunks": final_state.get("retrieved_chunks", []),
                },
            )
            _save_session_memory(request.session_id, final_state, request.query)
        else:
            log.info("No LangGraph checkpointer configured; using durable conversation store for chat history.")
            final_state = graph.invoke(initial_state)
            _save_session_memory(request.session_id, final_state, request.query)

        payload = {
            "answer_professional": final_state["answer_professional"],
            "answer_plain": final_state["answer_plain"],
            "citations": final_state["citations"],
            "disclaimer": final_state["disclaimer"],
            "tool_calls": final_state["tool_calls_log"],
            "blocked": final_state["input_blocked"] or final_state["output_blocked"],
            "turn_intents": final_state.get("turn_intents", []),
            "followup_routing": final_state.get("followup_routing", "retrieve"),
            "requires_clarification": final_state.get("requires_clarification", False),
            "missing_clinical_dimensions": final_state.get("missing_clinical_dimensions", []),
            "clarification_rationale": final_state.get("clarification_rationale"),
            "expected_clarification": final_state.get("expected_clarification"),
            "rag_trace": final_state.get("rag_trace", []),
            "token_usage": final_state.get("token_usage", {}),
            "external_search_snippets": final_state.get("external_search_snippets", []),
            "safety_warning": final_state.get("safety_warning"),
            "safety_explanation": final_state.get("safety_explanation"),
            "safety_title": final_state.get("safety_title"),
            "trace_id": trace_id,
        }
        log_event(
            log,
            "chat_request_completed",
            trace_id=trace_id,
            session_id=request.session_id,
            blocked=payload["blocked"],
            tool_calls=len(payload.get("tool_calls", [])),
            followup_routing=payload.get("followup_routing"),
            requires_clarification=payload.get("requires_clarification"),
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            detail = getattr(exc, "detail", {}) or {}
            technical_details = build_technical_details(
                raw_request,
                status_code=429,
                error_type=type(exc).__name__,
                detail=detail,
            )
            log_event(
                log,
                "chat_request_rate_limited",
                level="warning",
                trace_id=trace_id,
                session_id=request.session_id,
                retry_after_seconds=detail.get("retry_after_seconds"),
            )
            payload = {
                "answer_professional": detail.get("message", "Too many requests"),
                "answer_plain": "Please wait a moment and try again.",
                "citations": [],
                "disclaimer": "",
                "tool_calls": [],
                "blocked": True,
                "blocked_reason": "rate_limit",
                "retry_after_seconds": detail.get("retry_after_seconds"),
                "blocked_explanation_title": "Why?",
                "blocked_explanation": detail.get("reason", "You have reached usage protection limits."),
                "technical_title": "Technical details",
                "technical_details": technical_details,
                "requires_clarification": False,
                "missing_clinical_dimensions": [],
                "clarification_rationale": None,
                "expected_clarification": None,
                "rag_trace": [],
                "token_usage": {},
                "external_search_snippets": [],
                "safety_warning": None,
                "safety_explanation": None,
                "safety_title": None,
                "trace_id": trace_id,
            }
            body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
            headers = {"Cache-Control": "no-cache", "X-Trace-Id": trace_id}
            retry_after = getattr(exc, "headers", {}).get("Retry-After") if getattr(exc, "headers", None) else None
            if retry_after:
                headers["Retry-After"] = retry_after
            return Response(content=body, media_type="text/event-stream", headers=headers)
        technical_details = build_technical_details(
            raw_request,
            status_code=500,
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        log_event(
            log,
            "chat_request_failed",
            level="error",
            trace_id=trace_id,
            session_id=request.session_id,
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        payload = {
            "answer_professional": "A technical error interrupted the request.",
            "answer_plain": "Please try again. If the issue continues, open the technical details below and share the trace ID.",
            "citations": [],
            "disclaimer": "",
            "tool_calls": [],
            "blocked": False,
            "technical_title": "Technical details",
            "technical_details": technical_details,
            "requires_clarification": False,
            "missing_clinical_dimensions": [],
            "clarification_rationale": None,
            "expected_clarification": None,
            "rag_trace": [],
            "token_usage": {},
            "external_search_snippets": [],
            "safety_warning": None,
            "safety_explanation": None,
            "safety_title": None,
            "trace_id": trace_id,
        }

    body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    return Response(content=body, media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Trace-Id": trace_id})
