from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from src.api.auth import verify_api_key
from src.api.conversation_store import get_conversation_store
from src.api.export_utils import export_csv_bytes, export_json_bytes, export_pdf_bytes
from src.api.rate_limit import enforce_rate_limit, route_group_for_path

router = APIRouter()


class CreateConversationRequest(BaseModel):
    session_id: str | None = None
    title: str = "Neue Konversation"


@router.get("/conversations")
def list_conversations(request: Request):
    api_key = verify_api_key(request)
    enforce_rate_limit(request, api_key=api_key, route_group=route_group_for_path(request.url.path))
    store = get_conversation_store()
    return {"conversations": store.list_conversations()}


@router.post("/conversations")
def create_conversation(request: CreateConversationRequest, raw_request: Request):
    api_key = verify_api_key(raw_request)
    enforce_rate_limit(raw_request, api_key=api_key, route_group=route_group_for_path(raw_request.url.path))
    store = get_conversation_store()
    return store.create_conversation(request.session_id, request.title)


@router.delete("/conversations/{session_id}")
def delete_conversation(session_id: str, request: Request):
    api_key = verify_api_key(request)
    enforce_rate_limit(request, api_key=api_key, route_group=route_group_for_path(request.url.path))
    store = get_conversation_store()
    if not store.delete_conversation(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@router.get("/conversations/{session_id}/export")
def export_conversation(session_id: str, format: str, request: Request):
    api_key = verify_api_key(request)
    enforce_rate_limit(request, api_key=api_key, route_group=route_group_for_path(request.url.path))
    store = get_conversation_store()
    payload = store.export_conversation(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "-", payload.get("title", session_id)).strip("-") or session_id
    if format == "json":
        content = export_json_bytes(payload)
        media_type = "application/json"
        filename = f"{safe_base}.json"
    elif format == "csv":
        content = export_csv_bytes(payload)
        media_type = "text/csv; charset=utf-8"
        filename = f"{safe_base}.csv"
    elif format == "pdf":
        content = export_pdf_bytes(payload)
        media_type = "application/pdf"
        filename = f"{safe_base}.pdf"
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )