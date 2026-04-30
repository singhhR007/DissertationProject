from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.api_docs import PREDICTION_POST_ERROR_RESPONSES
from app.core.request_id import get_request_id
from app.core.security import require_bearer_token
from app.schemas.common import utc_now
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionResult,
    PredictionRequest,
    PredictionResponse,
)
from app.services.inference import predict_from_request
from app.services.model_registry import get_active_model_metadata


# Predictions router
router = APIRouter(
    prefix="/api/v1",
    tags=["predictions"],
)


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Single prediction",
    description=(
        "Submit a single structured log sequence or raw log text and receive "
        "an anomaly prediction response from the deployed model."
    ),
    dependencies=[Depends(require_bearer_token)],
    responses=PREDICTION_POST_ERROR_RESPONSES,
)
def create_prediction(request: Request, payload: PredictionRequest) -> PredictionResponse:
    """
    Return a prediction response for one validated request.
    """
    metadata = get_active_model_metadata()
    result = predict_from_request(payload)

    return PredictionResponse(
        request_id=get_request_id(request),
        prediction=result.prediction,
        risk_score=result.risk_score,
        threshold=result.threshold,
        score_type=metadata.score_type,
        model_version=result.model_version,
        processed_at=utc_now(),
        advisory=result.advisory,
    )


@router.post(
    "/predictions/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    description=(
        "Submit multiple structured log sequences or raw log text records and "
        "receive per-record prediction results from the deployed model."
    ),
    dependencies=[Depends(require_bearer_token)],
    responses=PREDICTION_POST_ERROR_RESPONSES,
)
def create_batch_prediction(
    request: Request,
    payload: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """
    Return a batch prediction response for validated batch input.
    """
    metadata = get_active_model_metadata()
    results: list[BatchPredictionResult] = []

    for index, record in enumerate(payload.records):
        result = predict_from_request(record)
        results.append(
            BatchPredictionResult(
                record_index=index,
                prediction=result.prediction,
                risk_score=result.risk_score,
                threshold=result.threshold,
            )
        )

    anomalous_count = sum(1 for item in results if item.prediction == "anomalous")

    return BatchPredictionResponse(
        request_id=get_request_id(request),
        total_records=len(results),
        anomalous_records_detected=anomalous_count,
        model_version=metadata.model_version,
        processed_at=utc_now(),
        results=results,
    )