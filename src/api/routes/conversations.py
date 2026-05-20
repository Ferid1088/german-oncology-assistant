from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.auth import verify_api_key
from src.api.conversation_store import get_conversation_store

router = APIRouter()


class CreateConversationRequest(BaseModel):
    session_id: str | None = None
    title: str = "Neue Konversation"


@router.get("/conversations", dependencies=[Depends(verify_api_key)])
def list_conversations():
    store = get_conversation_store()
    return {"conversations": store.list_conversations()}


@router.post("/conversations", dependencies=[Depends(verify_api_key)])
def create_conversation(request: CreateConversationRequest):
    store = get_conversation_store()
    return store.create_conversation(request.session_id, request.title)


@router.delete("/conversations/{session_id}", dependencies=[Depends(verify_api_key)])
def delete_conversation(session_id: str):
    store = get_conversation_store()
    if not store.delete_conversation(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}