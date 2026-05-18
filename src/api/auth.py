import os
from fastapi import Request, HTTPException

API_KEY = os.getenv("API_KEY", "dev-secret-key")


async def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
