from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.analytics_service import build_analytics_overview
from src.api.auth import verify_api_key
from src.api.conversation_store import get_conversation_store
from src.api.rate_limit import enforce_rate_limit, route_group_for_path

router = APIRouter()


@router.get("/analytics/overview")
def analytics_overview(request: Request, session_id: str | None = None):
    api_key = verify_api_key(request)
    enforce_rate_limit(request, api_key=api_key, route_group=route_group_for_path(request.url.path))
    store = get_conversation_store()
    return build_analytics_overview(store, session_id=session_id)