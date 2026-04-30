from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import Request, Response


# Request-ID helpers
def get_request_id(request: Request) -> UUID:
    """
    Return the request-scoped request ID as a UUID.

    Fall back to a new UUID if the request passed through code paths where the
    middleware was not reached or request.state was not populated.
    """
    raw_value = getattr(request.state, "request_id", None)

    if isinstance(raw_value, UUID):
        return raw_value

    if isinstance(raw_value, str):
        return UUID(raw_value)

    return uuid4()


# Request-ID middleware
# Each incoming request receives a server-generated request identifier.
# The ID is stored on request.state so that routes, exception handlers and
# future logging/observability code can all access the same value.
async def request_id_middleware(request: Request, call_next) -> Response:
    """
    Attach a unique server-generated request ID to the request lifecycle.
    """
    request_id = uuid4()
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = str(request_id)
    return response