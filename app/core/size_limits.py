from __future__ import annotations

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from app.core.request_id import get_request_id
from app.schemas.common import utc_now
from app.schemas.error import ErrorCode, ErrorObject, ErrorResponse


# Request-size limit configuration
SINGLE_PREDICTION_MAX_BYTES = 256 * 1024        # 256 KiB
BATCH_PREDICTION_MAX_BYTES = 2 * 1024 * 1024   # 2 MiB


def get_max_request_size(path: str) -> int | None:
    """
    Return the configured maximum request size for a given endpoint path.
    """
    if path == "/api/v1/predictions":
        return SINGLE_PREDICTION_MAX_BYTES
    if path == "/api/v1/predictions/batch":
        return BATCH_PREDICTION_MAX_BYTES
    return None


def build_payload_too_large_response(request: Request, limit_bytes: int) -> JSONResponse:
    """
    Return a standardized 413 error response.
    """
    payload = ErrorResponse(
        request_id=get_request_id(request),
        error=ErrorObject(
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            message=f"Request body exceeds the maximum allowed size of {limit_bytes} bytes.",
            details=[],
        ),
        processed_at=utc_now(),
    )

    response = JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content=payload.model_dump(mode="json"),
    )
    response.headers["X-Request-ID"] = str(payload.request_id)
    return response


# Request-size middleware

async def request_size_limit_middleware(request: Request, call_next) -> Response:
    """
    Enforce request-size limits for protected prediction endpoints.

    This middleware uses the fully received request body size for enforcement.
    That is sufficient for the artefact stage and keeps the behaviour explicit
    and testable inside the application.
    """
    max_size = get_max_request_size(request.url.path)

    # Only enforce limits for endpoints that have explicit request-size rules.
    if max_size is None:
        return await call_next(request)

    # Only POST endpoints carry prediction payloads in this contract.
    if request.method.upper() != "POST":
        return await call_next(request)

    body = await request.body()

    if len(body) > max_size:
        return build_payload_too_large_response(request, max_size)

    return await call_next(request)