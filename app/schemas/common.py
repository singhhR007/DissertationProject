from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field


class ClosedSchemaModel(BaseModel):
    """
    Shared base model for closed API schemas.

    Unknown fields must be rejected so that the runtime contract stays strict
    and versioned.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )


# Shared field aliases

SourceIdentifier = Annotated[str, Field(min_length=1, max_length=64)]
SequenceIdentifier = Annotated[str, Field(min_length=1, max_length=128)]
ProbabilityScore = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]


# Shared validation helpers

def validate_utc_z_timestamp_string(value: Any, *, field_name: str = "timestamp") -> Any:
    """
    Validate raw timestamp input before datetime parsing.

    Version 1 requires an ISO 8601 / RFC 3339 UTC timestamp in canonical Z form,
    for example:
        2026-04-03T12:30:00Z
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be provided as an ISO 8601 UTC string ending in 'Z'.")
    if not value.endswith("Z"):
        raise ValueError(f"{field_name} must be in UTC and must end with 'Z'.")
    return value


def validate_utc_datetime(value: datetime, *, field_name: str) -> datetime:
    """
    Ensure a datetime value is timezone-aware and in UTC.
    """
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must include timezone information.")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field_name} must be in UTC.")
    return value


def validate_json_object_size(
    value: dict[str, Any] | None,
    *,
    field_name: str,
    max_bytes: int,
) -> dict[str, Any] | None:
    """
    Validate that an optional JSON object stays below a byte-size limit when
    serialized as UTF-8 JSON.
    """
    if value is None:
        return value

    try:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable.") from exc

    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} must not exceed {max_bytes} bytes when serialized.")

    return value


def utc_now() -> datetime:
    """
    Return the current UTC time as a timezone-aware datetime.
    """
    return datetime.now(timezone.utc)