from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.schemas.common import (
    ClosedSchemaModel,
    ProbabilityScore,
    SourceIdentifier,
    validate_json_object_size,
    validate_utc_datetime,
    validate_utc_z_timestamp_string,
)
from app.schemas.telemetry import LogSequenceV1


# Shared type aliases for prediction-related schemas

PredictionLabel = Literal["normal", "anomalous"]
ScoreType = Literal["calibrated_anomalous_class_probability"]
RawTextFormat = Literal["auto", "hdfs", "openstack"]


# Request models

class PredictionRequest(ClosedSchemaModel):
    """
    Request schema for POST /api/v1/predictions.

    Version 1 keeps a single analytical telemetry type:
    - telemetry_type = "log_sequence"
    - telemetry_schema_version = "log_sequence_v1"

    However, the API accepts two input modes:
    - structured `log_sequence`
    - convenience `raw_log_text`, which is internally transformed into a
      normalized log sequence before inference

    Important:
    This schema represents the validated external runtime contract. It is not
    assumed to be identical to the final internal model feature vector.
    The deployed preprocessing pipeline may transform this validated request
    into the exact model input representation used for inference.

    Notes:
    - `source` is request-level metadata describing the producer-side or
      sequence-level origin of the payload.
    - `context` is optional and not used for inference in version 1.
    - Exactly one of `log_sequence` or `raw_log_text` must be provided.
    """

    timestamp: datetime = Field(
        description="UTC timestamp in ISO 8601 / RFC 3339 format ending in 'Z'."
    )
    source: SourceIdentifier = Field(
        description=(
            "Producer-side or sequence-level origin identifier for the payload, "
            "such as a node label, collector identifier, or upstream source ID."
        )
    )
    telemetry_type: Literal["log_sequence"] = Field(
        description='Fixed value for version 1. Must equal "log_sequence".'
    )
    telemetry_schema_version: Literal["log_sequence_v1"] = Field(
        description='Fixed value for version 1. Must equal "log_sequence_v1".'
    )
    log_sequence: LogSequenceV1 | None = Field(
        default=None,
        description=(
            "Structured log sequence input. Exactly one of `log_sequence` or "
            "`raw_log_text` must be provided."
        ),
    )
    raw_log_text: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Raw log text input. When provided, the text is internally parsed "
            "and normalized into a log sequence before inference. Exactly one "
            "of `log_sequence` or `raw_log_text` must be provided."
        ),
    )
    raw_text_format: RawTextFormat | None = Field(
        default=None,
        description=(
            "Optional parsing hint used only with `raw_log_text`. "
            "Allowed values: `auto`, `hdfs`, `openstack`."
        ),
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional JSON object for contextual metadata only. "
            "Not used for inference in version 1."
        ),
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp_raw(cls, value: Any) -> Any:
        """
        Require canonical UTC timestamp input in Z form before parsing.
        """
        return validate_utc_z_timestamp_string(value, field_name="timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_utc(cls, value: datetime) -> datetime:
        """
        Ensure the parsed timestamp is timezone-aware and normalized to UTC.
        """
        return validate_utc_datetime(value, field_name="timestamp")

    @field_validator("raw_log_text")
    @classmethod
    def validate_raw_log_text(cls, value: str | None) -> str | None:
        """
        Ensure raw log text contains at least one non-empty line and normalize
        line endings for downstream preprocessing.
        """
        if value is None:
            return None

        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        if not any(line.strip() for line in normalized.split("\n")):
            raise ValueError("raw_log_text must contain at least one non-empty log line.")

        return normalized

    @field_validator("context")
    @classmethod
    def validate_context(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Ensure optional context is JSON-serializable and respects the
        contract-defined size limit.
        """
        return validate_json_object_size(value, field_name="context", max_bytes=4096)

    @model_validator(mode="after")
    def validate_input_mode(self) -> "PredictionRequest":
        """
        Enforce that exactly one input mode is used per request.
        """
        has_log_sequence = self.log_sequence is not None
        has_raw_log_text = self.raw_log_text is not None

        if has_log_sequence == has_raw_log_text:
            raise ValueError(
                "Exactly one of log_sequence or raw_log_text must be provided."
            )

        if not has_raw_log_text and self.raw_text_format is not None:
            raise ValueError(
                "raw_text_format may be provided only when raw_log_text is used."
            )

        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2026-04-03T12:30:00Z",
                "source": "hdfs-node-01",
                "telemetry_type": "log_sequence",
                "telemetry_schema_version": "log_sequence_v1",
                "log_sequence": {
                    "sequence_id": "blk_-1608999687919862906",
                    "events": [
                        {
                            "message": (
                                "Receiving block blk_-1608999687919862906 src: "
                                "/10.250.19.102:54106 dest: /10.250.19.102:50010"
                            ),
                            "component": "DataNode",
                            "severity": "info",
                            "host": "10.250.19.102",
                            "service": "hdfs",
                        },
                        {
                            "message": (
                                "PacketResponder 1 for block "
                                "blk_-1608999687919862906 terminating"
                            ),
                            "component": "DataNode",
                            "severity": "info",
                            "host": "10.250.19.102",
                            "service": "hdfs",
                        },
                    ],
                },
                "context": {
                    "environment": "lab",
                },
            }
        }
    )


class BatchPredictionRequest(ClosedSchemaModel):
    """
    Request schema for POST /api/v1/predictions/batch.

    Version 1 uses all-or-nothing validation:
    every record must be valid, otherwise the entire batch request is rejected.

    The maximum batch size for version 1 is 100 records.
    Each record may use either structured `log_sequence` input or `raw_log_text`
    input, but not both at the same time.
    """

    records: list[PredictionRequest] = Field(
        min_length=1,
        max_length=100,
        description="Batch of 1 to 100 prediction records.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "records": [
                    {
                        "timestamp": "2026-04-03T12:30:00Z",
                        "source": "hdfs-node-01",
                        "telemetry_type": "log_sequence",
                        "telemetry_schema_version": "log_sequence_v1",
                        "log_sequence": {
                            "sequence_id": "blk_-1608999687919862906",
                            "events": [
                                {
                                    "message": (
                                        "Receiving block blk_-1608999687919862906 src: "
                                        "/10.250.19.102:54106 dest: /10.250.19.102:50010"
                                    ),
                                    "component": "DataNode",
                                    "severity": "info",
                                    "host": "10.250.19.102",
                                    "service": "hdfs",
                                }
                            ],
                        },
                    }
                ]
            }
        }
    )


# Response models

class PredictionResponse(ClosedSchemaModel):
    """
    Success response schema for POST /api/v1/predictions.

    This response returns a decision-support result, not proof of maliciousness.
    The contract defines `risk_score` in version 1 as a calibrated anomalous-
    class probability in the range [0, 1].
    """

    request_id: UUID = Field(
        description="Server-generated unique request identifier."
    )
    prediction: PredictionLabel = Field(
        description='Prediction label. Must be either "normal" or "anomalous".'
    )
    risk_score: ProbabilityScore = Field(
        description="Calibrated anomalous-class probability in the range [0,1]."
    )
    threshold: ProbabilityScore = Field(
        description="Fixed deployed decision threshold used for classification."
    )
    score_type: ScoreType = Field(
        default="calibrated_anomalous_class_probability",
        description="Fixed semantics of risk_score in version 1.",
    )
    model_version: str = Field(
        min_length=1,
        max_length=32,
        description="Version of the deployed model.",
    )
    processed_at: datetime = Field(
        description="UTC timestamp indicating when the response was generated."
    )
    advisory: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Optional human-readable guidance message.",
    )

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at_utc(cls, value: datetime) -> datetime:
        """
        Ensure the response timestamp is timezone-aware and in UTC.
        """
        return validate_utc_datetime(value, field_name="processed_at")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "0f3d8db8-1f6c-4bdf-8d92-11e0f0e2f91a",
                "prediction": "anomalous",
                "risk_score": 0.83,
                "threshold": 0.70,
                "score_type": "calibrated_anomalous_class_probability",
                "model_version": "v1.0.0",
                "processed_at": "2026-04-03T12:30:02Z",
                "advisory": (
                    "Model output indicates anomalous behaviour. "
                    "Human review is recommended."
                ),
            }
        }
    )


class BatchPredictionResult(ClosedSchemaModel):
    """
    Per-record prediction result embedded in batch prediction responses.
    """

    record_index: int = Field(
        ge=0,
        description="Zero-based position of the record in the submitted batch.",
    )
    prediction: PredictionLabel = Field(
        description='Prediction label. Must be either "normal" or "anomalous".'
    )
    risk_score: ProbabilityScore = Field(
        description="Calibrated anomalous-class probability in the range [0,1]."
    )
    threshold: ProbabilityScore = Field(
        description="Fixed deployed decision threshold used for classification."
    )


class BatchPredictionResponse(ClosedSchemaModel):
    """
    Success response schema for POST /api/v1/predictions/batch.

    The response returns a list of per-record predictions plus minimal batch-
    level summary information.

    This model also enforces internal consistency between:
    - total_records
    - anomalous_records_detected
    - results
    """

    request_id: UUID = Field(
        description="Server-generated unique request identifier."
    )
    total_records: int = Field(
        ge=1,
        le=100,
        description="Total number of submitted sequence records processed in the batch.",
    )
    anomalous_records_detected: int = Field(
        ge=0,
        description="Number of submitted sequence records classified as anomalous.",
    )
    model_version: str = Field(
        min_length=1,
        max_length=32,
        description="Version of the deployed model.",
    )
    processed_at: datetime = Field(
        description="UTC timestamp indicating when the batch response was generated."
    )
    results: list[BatchPredictionResult] = Field(
        min_length=1,
        max_length=100,
        description="Per-record prediction results.",
    )

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at_utc(cls, value: datetime) -> datetime:
        """
        Ensure the response timestamp is timezone-aware and in UTC.
        """
        return validate_utc_datetime(value, field_name="processed_at")

    @model_validator(mode="after")
    def validate_batch_consistency(self) -> "BatchPredictionResponse":
        """
        Enforce semantic consistency across batch-level summary fields.
        """
        if len(self.results) != self.total_records:
            raise ValueError("results length must equal total_records.")

        anomalous_count = sum(
            1 for item in self.results if item.prediction == "anomalous"
        )
        if self.anomalous_records_detected != anomalous_count:
            raise ValueError(
                "anomalous_records_detected must match the number of anomalous results."
            )

        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "81ed9fb6-7fd6-4916-a953-24f4e2d2fd75",
                "total_records": 2,
                "anomalous_records_detected": 1,
                "model_version": "v1.0.0",
                "processed_at": "2026-04-03T12:31:05Z",
                "results": [
                    {
                        "record_index": 0,
                        "prediction": "anomalous",
                        "risk_score": 0.83,
                        "threshold": 0.70,
                    },
                    {
                        "record_index": 1,
                        "prediction": "normal",
                        "risk_score": 0.19,
                        "threshold": 0.70,
                    },
                ],
            }
        }
    )