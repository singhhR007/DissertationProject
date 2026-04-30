from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.error import ErrorCode, ErrorDetail, ErrorObject, ErrorResponse
from app.schemas.common import utc_now


# Helper functions

def get_request_id(request: Request) -> str:
    """
    Return the request-scoped request ID if available.
    Fall back to a new UUID if middleware was not reached.
    """
    return getattr(request.state, "request_id", str(uuid4()))


def map_http_status_to_error_code(status_code: int) -> ErrorCode:
    """
    Map HTTP status codes to stable machine-readable error codes.
    """
    if status_code == status.HTTP_401_UNAUTHORIZED:
        return ErrorCode.UNAUTHORIZED
    if status_code == status.HTTP_403_FORBIDDEN:
        return ErrorCode.FORBIDDEN
    if status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE:
        return ErrorCode.PAYLOAD_TOO_LARGE
    if status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE:
        return ErrorCode.UNSUPPORTED_MEDIA_TYPE
    if status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return ErrorCode.SERVICE_UNAVAILABLE

    # For 400/422 and similar client-side payload issues, VALIDATION_ERROR is
    # the closest current stable error code in the schema.
    if status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    }:
        return ErrorCode.VALIDATION_ERROR

    return ErrorCode.INTERNAL_ERROR


def format_validation_location(loc: tuple[Any, ...]) -> str:
    """
    Convert FastAPI/Pydantic validation locations into compact field paths.

    Examples:
    - ('body', 'records', 0, 'log_sequence', 'events', 0, 'message')
      -> 'records[0].log_sequence.events[0].message'
    - ('query', 'limit') -> 'query.limit'
    """
    if not loc:
        return "request"

    items = list(loc)

    prefix = ""
    if items[0] in {"body", "query", "path", "header"}:
        first = items.pop(0)
        if first != "body":
            prefix = first

    path = ""
    for item in items:
        if isinstance(item, int):
            path += f"[{item}]"
        else:
            path += f".{item}" if path else str(item)

    if prefix and path:
        return f"{prefix}.{path}"
    if prefix:
        return prefix
    return path or "request"


def build_error_response(
    *,
    request: Request,
    status_code: int,
    code: ErrorCode,
    message: str,
    details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """
    Build a JSONResponse using the standard ErrorResponse schema.
    """
    payload = ErrorResponse(
        request_id=get_request_id(request),
        error=ErrorObject(
            code=code,
            message=message,
            details=details or [],
        ),
        processed_at=utc_now(),
    )

    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


# Exception handlers

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle expected HTTPExceptions such as authentication failures.
    """
    code = map_http_status_to_error_code(exc.status_code)

    if isinstance(exc.detail, str) and exc.detail.strip():
        message = exc.detail
    else:
        message = "Request failed."

    return build_error_response(
        request=request,
        status_code=exc.status_code,
        code=code,
        message=message,
        headers=exc.headers,
    )


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle request validation failures and return the standard error schema.
    """
    details: list[ErrorDetail] = []

    for err in exc.errors()[:10]:
        field_path = format_validation_location(tuple(err.get("loc", ())))
        issue = err.get("msg", "Validation error.")
        details.append(
            ErrorDetail(
                field=field_path,
                issue=issue,
            )
        )

    return build_error_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code=ErrorCode.VALIDATION_ERROR,
        message="Request payload failed schema validation.",
        details=details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected internal errors without leaking implementation details.
    """
    return build_error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code=ErrorCode.INTERNAL_ERROR,
        message="An internal server error occurred.",
    )


# Registration helper

def register_exception_handlers(app: FastAPI) -> None:
    """
    Register centralized exception handlers for the API.
    """
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)