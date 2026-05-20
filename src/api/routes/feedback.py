import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel
from src.api.auth import verify_api_key
from src.api.rate_limit import enforce_rate_limit, route_group_for_path

router = APIRouter()
log = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    rating: int   # 1 = thumbs up, -1 = thumbs down
    comment: str = ""


@router.post("/feedback")
async def feedback(request: FeedbackRequest, raw_request: Request):
    api_key = verify_api_key(raw_request)
    enforce_rate_limit(raw_request, api_key=api_key, route_group=route_group_for_path(raw_request.url.path))
    log.info("Feedback: session=%s rating=%d", request.session_id, request.rating)
    return {"status": "ok"}
