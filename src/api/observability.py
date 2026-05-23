"""Observability utilities for the FastAPI application.

Provides two entry points called from ``main.py``:
- ``configure_logging()`` — sets up ``basicConfig`` with a level read from ``LOG_LEVEL``
  (default INFO) and a plain ``%(message)s`` format (all log lines are JSON objects).
- ``configure_observability(app)`` — idempotent guard that attaches three pieces to the
  running FastAPI application:
  1. ``trace_middleware`` — assigns a unique trace ID to every request, logs completion
     with method, path, status code, and duration.
  2. ``http_exception_handler`` — normalises FastAPI ``HTTPException`` (including rate
     limit 429s and auth 401s) into a consistent JSON envelope with ``trace_id``.
  3. ``validation_exception_handler`` — handles Pydantic validation errors (422) the
     same way.
  4. ``unhandled_exception_handler`` — catches any other ``Exception`` and returns 500
     without leaking a stack trace to the client.
"""

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
    """Coerce an arbitrary value to a JSON-serialisable form using ``default=str``."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def configure_logging() -> None:
    """Configure the root logger from the ``LOG_LEVEL`` environment variable.

    Falls back to INFO when the variable is absent or contains an unrecognised level name.
    All log output uses a plain ``%(message)s`` format because every log line is a JSON
    object emitted via ``log_event``.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def log_event(logger: logging.Logger, event: str, level: str = "info", **fields) -> None:
    """Emit a structured JSON log line via ``logger``.

    Args:
        logger: The logger instance to use (typically ``logging.getLogger("src.api")``).
        event: Short event name, e.g. ``"request_complete"`` or ``"http_exception"``.
        level: Log level string matching a ``logging.Logger`` method (default ``"info"``).
        **fields: Additional key-value pairs merged into the log payload.
    """
    payload = {"event": event, **fields}
    log_method = getattr(logger, level, logger.info)
    log_method(json.dumps(payload, ensure_ascii=False, default=str))


def get_trace_id(request: Request) -> str:
    """Return the trace ID for a request, assigning one if not already set.

    Checks ``request.state.trace_id`` first (set by earlier middleware), then the
    ``X-Trace-Id`` request header, then generates a new 12-hex-char UUID fragment.

    Args:
        request: The incoming FastAPI request object.

    Returns:
        A stable trace ID string for the lifetime of this request.
    """
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
    """Build the ``technical_details`` block included in all error responses.

    Args:
        request: The FastAPI request that triggered the error.
        status_code: HTTP status code of the error response.
        error_type: Exception class name (e.g. ``"HTTPException"``).
        detail: Raw exception detail value; serialised via ``_json_safe``.

    Returns:
        A dict with ``trace_id``, ``path``, ``method``, ``status_code``,
        ``error_type``, and ``detail`` fields.
    """
    return {
        "trace_id": get_trace_id(request),
        "path": request.url.path,
        "method": request.method,
        "status_code": status_code,
        "error_type": error_type,
        "detail": _json_safe(detail),
    }


def _friendly_message_for_status(status_code: int) -> str:
    """Map an HTTP status code to a user-facing error message string."""
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
    """Attach middleware and exception handlers to the FastAPI app.

    Idempotent: subsequent calls are no-ops, checked via ``app.state.observability_configured``.
    Must be called after all routers are registered so the middleware sees all routes.

    Args:
        app: The FastAPI application instance.
    """
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