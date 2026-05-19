import json
from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from src.api.auth import verify_api_key
from src.graph.graph import build_graph
from src.graph.state import RAGState

router = APIRouter()
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


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
        initial_state = RAGState(
            user_query=request.query,
            session_id=request.session_id,
            rewritten_query="",
            metadata_filters={k: v for k, v in {
                "guideline_id": request.guideline_id,
                "grade": request.grade,
            }.items() if v},
            intent="",
            retrieved_chunks=[],
            confidence=0.0,
            answer_professional="",
            answer_plain="",
            citations=[],
            disclaimer="",
            input_blocked=False,
            input_block_reason="",
            output_blocked=False,
            tool_calls_log=[],
            messages=[],
        )

        final_state = graph.invoke(initial_state)

        payload = {
            "answer_professional": final_state["answer_professional"],
            "answer_plain": final_state["answer_plain"],
            "citations": final_state["citations"],
            "disclaimer": final_state["disclaimer"],
            "tool_calls": final_state["tool_calls_log"],
            "blocked": final_state["input_blocked"] or final_state["output_blocked"],
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
