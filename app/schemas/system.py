from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from app.schemas.common import ClosedSchemaModel, ProbabilityScore, validate_utc_datetime


class HealthResponse(ClosedSchemaModel):
    """
    Response schema for GET /api/v1/health.

    This endpoint is intentionally minimal. It is public and exposes only basic
    readiness/liveness information, not sensitive model metadata.
    """

    status: Literal["healthy"] = Field(
        default="healthy",
        description='Public health status. Fixed to "healthy" for a successful response.',
    )
    ready: Literal[True] = Field(
        default=True,
        description="Indicates that the API is ready to serve requests.",
    )
    api_version: str = Field(
        min_length=1,
        max_length=32,
        description="Version of the deployed API.",
        examples=["1.0.0"],
    )
    processed_at: datetime = Field(
        description="UTC timestamp indicating when the health response was generated.",
        examples=["2026-04-03T12:31:10Z"],
    )

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at_utc(cls, value: datetime) -> datetime:
        return validate_utc_datetime(value, field_name="processed_at")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "ready": True,
                "api_version": "1.0.0",
                "processed_at": "2026-04-03T12:31:10Z"
            }
        }
    )


class ModelInfoResponse(ClosedSchemaModel):
    """
    Response schema for GET /api/v1/model/info.

    This is a protected technical transparency endpoint used for controlled
    inspection, artefact evaluation and debugging. It is not intended as a
    public-facing business endpoint.
    """

    model_name: str = Field(
        min_length=1,
        max_length=128,
        description="Human-readable identifier of the deployed model.",
        examples=["baseline_log_anomaly_classifier"],
    )
    model_version: str = Field(
        min_length=1,
        max_length=32,
        description="Version of the deployed model.",
        examples=["v1.0.0"],
    )
    primary_dataset: Literal["HDFS"] = Field(
        default="HDFS",
        description="Primary dataset used for model development and threshold selection.",
    )
    secondary_offline_benchmark: Literal["OpenStack"] = Field(
        default="OpenStack",
        description="Secondary offline benchmark used only for comparative evaluation.",
    )
    telemetry_type: Literal["log_sequence"] = Field(
        default="log_sequence",
        description='Supported runtime telemetry type for version 1.',
    )
    telemetry_schema_version: Literal["log_sequence_v1"] = Field(
        default="log_sequence_v1",
        description='Supported runtime telemetry schema version for version 1.',
    )
    score_type: Literal["calibrated_anomalous_class_probability"] = Field(
        default="calibrated_anomalous_class_probability",
        description="Fixed semantics of risk_score in version 1.",
    )
    threshold_selection_objective: Literal["max_f1_on_primary_validation_split"] = Field(
        default="max_f1_on_primary_validation_split",
        description="Declared objective used to select the deployed threshold.",
    )
    threshold: ProbabilityScore = Field(
        description="Fixed deployed decision threshold.",
        examples=[0.70],
    )
    last_updated: datetime = Field(
        description="UTC timestamp indicating when the deployed model metadata was last updated.",
        examples=["2026-04-03T10:00:00Z"],
    )

    @field_validator("last_updated")
    @classmethod
    def validate_last_updated_utc(cls, value: datetime) -> datetime:
        return validate_utc_datetime(value, field_name="last_updated")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "baseline_log_anomaly_classifier",
                "model_version": "v1.0.0",
                "primary_dataset": "HDFS",
                "secondary_offline_benchmark": "OpenStack",
                "telemetry_type": "log_sequence",
                "telemetry_schema_version": "log_sequence_v1",
                "score_type": "calibrated_anomalous_class_probability",
                "threshold_selection_objective": "max_f1_on_primary_validation_split",
                "threshold": 0.70,
                "last_updated": "2026-04-03T10:00:00Z"
            }
        }
    )