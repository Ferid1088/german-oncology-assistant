from __future__ import annotations

import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def log_event(logger: logging.Logger, event: str, level: str = "info", **fields) -> None:
    payload = {"event": event, **fields}
    log_method = getattr(logger, level, logger.info)
    log_method(json.dumps(payload, ensure_ascii=False, default=str))


def get_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if trace_id:
        return str(trace_id)

    trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:12]
    request.state.trace_id = trace_id
    return trace_id


def build_technical_details(
    request: Request,
    *,
    status_code: int,
    error_type: str,
    detail,
) -> dict:
    return {
        "trace_id": get_trace_id(request),
        "path": request.url.path,
        "method": request.method,
        "status_code": status_code,
        "error_type": error_type,
        "detail": _json_safe(detail),
    }


def _friendly_message_for_status(status_code: int) -> str:
    if status_code == 401:
        return "Authentication failed."
    if status_code == 404:
        return "The requested resource was not found."
    if status_code == 422:
        return "The request data is invalid."
    if status_code == 429:
        return "Too many requests."
    return "The request could not be completed."


def configure_observability(app: FastAPI) -> None:
    if getattr(app.state, "observability_configured", False):
        return

    logger = logging.getLogger("src.api")

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        trace_id = get_trace_id(request)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            log_event(
                logger,
                "request_exception",
                level="error",
                trace_id=trace_id,
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Trace-Id"] = trace_id
        log_event(
            logger,
            "request_complete",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        reason = detail.get("reason") if isinstance(detail, dict) else None
        retry_after = detail.get("retry_after_seconds") if isinstance(detail, dict) else None
        message = detail.get("message") if isinstance(detail, dict) else None
        technical_details = build_technical_details(
            request,
            status_code=exc.status_code,
            error_type=type(exc).__name__,
            detail=detail,
        )
        trace_id = technical_details["trace_id"]
        log_event(
            logger,
            "http_exception",
            level="warning",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status_code=exc.status_code,
            detail=detail,
        )

        payload = {
            "message": message or _friendly_message_for_status(exc.status_code),
            "trace_id": trace_id,
            "technical_title": "Technical details",
            "technical_details": technical_details,
        }
        if reason:
            payload["reason"] = reason
        if retry_after is not None:
            payload["retry_after_seconds"] = retry_after

        headers = dict(exc.headers or {})
        headers["X-Trace-Id"] = trace_id
        return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        safe_errors = _json_safe(exc.errors())
        technical_details = build_technical_details(
            request,
            status_code=422,
            error_type=type(exc).__name__,
            detail=safe_errors,
        )
        trace_id = technical_details["trace_id"]
        log_event(
            logger,
            "validation_exception",
            level="warning",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            detail=safe_errors,
        )

        return JSONResponse(
            status_code=422,
            content={
                "message": "The request data is invalid.",
                "trace_id": trace_id,
                "technical_title": "Technical details",
                "technical_details": technical_details,
            },
            headers={"X-Trace-Id": trace_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        technical_details = build_technical_details(
            request,
            status_code=500,
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        trace_id = technical_details["trace_id"]
        log_event(
            logger,
            "unhandled_exception",
            level="error",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            error_type=type(exc).__name__,
            detail=str(exc),
        )

        return JSONResponse(
            status_code=500,
            content={
                "message": "An internal error occurred.",
                "trace_id": trace_id,
                "technical_title": "Technical details",
                "technical_details": technical_details,
            },
            headers={"X-Trace-Id": trace_id},
        )

    app.state.observability_configured = True