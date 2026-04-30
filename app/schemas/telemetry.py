from __future__ import annotations

from enum import Enum

from pydantic import ConfigDict, Field

from app.schemas.common import ClosedSchemaModel, SequenceIdentifier


class LogSeverity(str, Enum):
    """
    Common severity levels supported by the log-sequence runtime schema.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogEventV1(ClosedSchemaModel):
    """
    Closed event schema used inside log_sequence_v1.

    Each event represents one validated log line plus optional structured
    metadata that may be extracted or provided by upstream systems.
    """

    message: str = Field(
        min_length=1,
        max_length=4096,
        description="Raw log message content."
    )
    component: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional log component or subsystem name."
    )
    severity: LogSeverity | None = Field(
        default=None,
        description="Optional log severity level."
    )
    host: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Optional host identifier associated with the event."
    )
    service: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional service or application name."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": (
                    "Receiving block blk_-1608999687919862906 src: "
                    "/10.250.19.102:54106 dest: /10.250.19.102:50010"
                ),
                "component": "DataNode",
                "severity": "info",
                "host": "10.250.19.102",
                "service": "hdfs"
            }
        }
    )


class LogSequenceV1(ClosedSchemaModel):
    """
    Closed runtime telemetry schema for version 1.

    This schema represents a structured log sequence suitable for log-based
    anomaly detection. It is intentionally general enough to support HDFS and
    OpenStack-style sequence payloads without claiming universal log support.
    """

    sequence_id: SequenceIdentifier = Field(
        description=(
            "Logical sequence identifier used by the preprocessing pipeline, "
            "such as a block ID, session ID, request ID, or parser-defined key."
        )
    )
    events: list[LogEventV1] = Field(
        min_length=1,
        max_length=500,
        description="Ordered list of validated log events belonging to one logical sequence."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                        "service": "hdfs"
                    },
                    {
                        "message": "PacketResponder 1 for block blk_-1608999687919862906 terminating",
                        "component": "DataNode",
                        "severity": "info",
                        "host": "10.250.19.102",
                        "service": "hdfs"
                    }
                ]
            }
        }
    )