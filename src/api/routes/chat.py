import json
import os
import logging
from threading import Lock
from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from src.api.auth import verify_api_key
from src.graph.graph import build_graph
from src.graph.checkpointing import build_checkpointer
from src.graph.state import RAGState

router = APIRouter()
_graph = None
log = logging.getLogger(__name__)
_memory_lock = Lock()
_session_memory: dict[str, dict] = {}
DEFAULT_USER_ROLE = os.getenv("DEFAULT_USER_ROLE", "user")
DEFAULT_ALLOWED_SOURCES = ["guidelines"]


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=build_checkpointer())
    return _graph


def _supports_checkpointing(graph) -> bool:
    return getattr(graph, "checkpointer", None) is not None


def _load_session_memory(session_id: str) -> dict:
    with _memory_lock:
        memory = _session_memory.get(session_id)
        if not memory:
            return {
                "messages": [],
                "prior_answer_professional": "",
                "prior_answer_plain": "",
                "prior_citations": [],
                "prior_retrieved_chunks": [],
            }
        return {
            "messages": list(memory.get("messages", [])),
            "prior_answer_professional": memory.get("prior_answer_professional", ""),
            "prior_answer_plain": memory.get("prior_answer_plain", ""),
            "prior_citations": list(memory.get("prior_citations", [])),
            "prior_retrieved_chunks": list(memory.get("prior_retrieved_chunks", [])),
        }


def _save_session_memory(session_id: str, final_state: dict, user_query: str) -> None:
    with _memory_lock:
        previous = _session_memory.get(session_id, {})
        messages = list(previous.get("messages", []))
        messages.append(HumanMessage(content=user_query))
        messages.append(AIMessage(content=final_state["answer_professional"]))
        _session_memory[session_id] = {
            "messages": messages[-12:],
            "prior_answer_professional": final_state.get("answer_professional", ""),
            "prior_answer_plain": final_state.get("answer_plain", ""),
            "prior_citations": list(final_state.get("citations", [])),
            "prior_retrieved_chunks": list(final_state.get("retrieved_chunks", [])),
        }


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    guideline_id: str = ""
    grade: str = ""


@router.post("/chat", dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest):
    # Sync endpoint — FastAPI runs this in a thread pool, so blocking LLM calls are safe.
    try:
        graph = get_graph()
        use_checkpoint = _supports_checkpointing(graph)
        session_memory = _load_session_memory(request.session_id) if not use_checkpoint else {
            "messages": [HumanMessage(content=request.query)],
            "prior_answer_professional": "",
            "prior_answer_plain": "",
            "prior_citations": [],
            "prior_retrieved_chunks": [],
        }
        input_messages = list(session_memory.get("messages", []))
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
            turn_intents=[],
            followup_routing="retrieve",
            prior_answer_professional=session_memory.get("prior_answer_professional", ""),
            prior_answer_plain=session_memory.get("prior_answer_plain", ""),
            prior_citations=session_memory.get("prior_citations", []),
            prior_retrieved_chunks=session_memory.get("prior_retrieved_chunks", []),
            messages=input_messages,
        )

        if use_checkpoint:
            config = {"configurable": {"thread_id": request.session_id}}
            final_state = graph.invoke(initial_state, config=config)
            graph.update_state(
                config,
                {
                    "messages": [AIMessage(content=final_state["answer_professional"])],
                    "prior_answer_professional": final_state["answer_professional"],
                    "prior_answer_plain": final_state["answer_plain"],
                    "prior_citations": final_state["citations"],
                    "prior_retrieved_chunks": final_state.get("retrieved_chunks", []),
                },
            )
        else:
            log.info("No checkpointer configured; running chat request without persisted conversation memory.")
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
        }
    except Exception as exc:
        payload = {
            "answer_professional": f"Interner Fehler: {exc}",
            "answer_plain": "Es ist ein Fehler aufgetreten. Bitte prüfen Sie die Serverkonfiguration.",
            "citations": [],
            "disclaimer": "",
            "tool_calls": [],
            "blocked": False,
        }

    body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    return Response(content=body, media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"})
