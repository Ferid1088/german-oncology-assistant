import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.api.auth import verify_api_key

router = APIRouter()
log = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    rating: int   # 1 = thumbs up, -1 = thumbs down
    comment: str = ""


@router.post("/feedback", dependencies=[Depends(verify_api_key)])
async def feedback(request: FeedbackRequest):
    log.info("Feedback: session=%s rating=%d", request.session_id, request.rating)
    return {"status": "ok"}
