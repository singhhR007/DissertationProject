from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator

from app.schemas.common import ClosedSchemaModel, validate_utc_datetime


class ErrorCode(str, Enum):
    """
    Stable machine-readable error codes supported by version 1.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class ErrorDetail(ClosedSchemaModel):
    """
    Field-level validation detail used in structured error responses.

    Only field paths and validation issues may be exposed. Internal stack
    traces, file paths, secret values and implementation details must never be
    returned to the client.
    """

    field: str = Field(
        min_length=1,
        max_length=256,
        description="Field path associated with the validation or processing issue.",
        examples=["records[0].log_sequence.events[0].message"],
    )
    issue: str = Field(
        min_length=1,
        max_length=512,
        description="Human-readable explanation of the issue.",
        examples=["Field is required."],
    )


class ErrorObject(ClosedSchemaModel):
    """
    Core error payload embedded in all API error responses.
    """

    code: ErrorCode = Field(
        description="Stable machine-readable error code.",
        examples=["VALIDATION_ERROR"],
    )
    message: str = Field(
        min_length=1,
        max_length=512,
        description="Human-readable summary of the error.",
        examples=["Request payload failed schema validation."],
    )
    details: list[ErrorDetail] = Field(
        default_factory=list,
        max_length=10,
        description="Optional list of field-level validation details. Maximum 10 items.",
    )


class ErrorResponse(ClosedSchemaModel):
    """
    Standard error response schema for the API.

    This structure is used consistently across authentication failures,
    validation failures and sanitised internal failures.
    """

    request_id: UUID = Field(
        description="Server-generated request identifier for traceability.",
        examples=["c60eb8a9-d54c-4c97-a273-4d3b9d8a8d3d"],
    )
    error: ErrorObject
    processed_at: datetime = Field(
        description="UTC timestamp indicating when the error response was generated.",
        examples=["2026-04-03T12:31:30Z"],
    )

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at_utc(cls, value: datetime) -> datetime:
        """
        Ensure processed_at is timezone-aware and in UTC.
        """
        return validate_utc_datetime(value, field_name="processed_at")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "c60eb8a9-d54c-4c97-a273-4d3b9d8a8d3d",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request payload failed schema validation.",
                    "details": [
                        {
                            "field": "records[0].log_sequence.events[0].message",
                            "issue": "Field is required."
                        }
                    ]
                },
                "processed_at": "2026-04-03T12:31:30Z"
            }
        }
    )