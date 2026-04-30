from __future__ import annotations

from fastapi import APIRouter

from app.schemas.common import utc_now
from app.schemas.system import HealthResponse


# System router
# This router contains operational/system-level endpoints.
# In version 1, the first public endpoint is the minimal health check defined
# by the API contract.
router = APIRouter(
    prefix="/api/v1",
    tags=["system"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Public minimal readiness/liveness check. "
        "This endpoint intentionally exposes only basic API health information."
    ),
)
def get_health() -> HealthResponse:
    """
    Return the public health status of the API.
    """
    return HealthResponse(
        api_version="1.0.0",
        processed_at=utc_now(),
    )