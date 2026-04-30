from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.api_docs import PROTECTED_GET_ERROR_RESPONSES
from app.core.security import require_bearer_token
from app.schemas.system import ModelInfoResponse
from app.services.model_registry import get_active_model_metadata


# Model router
router = APIRouter(
    prefix="/api/v1",
    tags=["model"],
)


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description=(
        "Protected technical transparency endpoint for model metadata. "
        "This endpoint is intended for controlled inspection and artefact "
        "evaluation rather than public business use."
    ),
    dependencies=[Depends(require_bearer_token)],
    responses=PROTECTED_GET_ERROR_RESPONSES,
)
def get_model_info() -> ModelInfoResponse:
    """
    Return metadata about the currently deployed model.
    """
    metadata = get_active_model_metadata()

    return ModelInfoResponse(
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        primary_dataset=metadata.primary_dataset,
        secondary_offline_benchmark=metadata.secondary_offline_benchmark,
        telemetry_type=metadata.telemetry_type,
        telemetry_schema_version=metadata.telemetry_schema_version,
        score_type=metadata.score_type,
        threshold_selection_objective=metadata.threshold_selection_objective,
        threshold=metadata.threshold,
        last_updated=metadata.last_updated,
    )