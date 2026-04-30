from __future__ import annotations

from app.schemas.prediction import PredictionRequest
from app.services.preprocessing import (
    normalize_prediction_request,
    normalize_raw_log_text,
)


def make_raw_hdfs_request() -> PredictionRequest:
    return PredictionRequest(
        timestamp="2026-04-03T12:30:00Z",
        source="hdfs-node-01",
        telemetry_type="log_sequence",
        telemetry_schema_version="log_sequence_v1",
        raw_log_text=(
            "081109 203615 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 src: "
            "/10.250.19.102:54106 dest: /10.250.19.102:50010\n"
            "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
            "PacketResponder 1 for block blk_-1608999687919862906 terminating"
        ),
        raw_text_format="hdfs",
        context={"environment": "lab"},
    )


def test_normalize_raw_log_text_hdfs_builds_sequence() -> None:
    sequence = normalize_raw_log_text(
        (
            "081109 203615 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 src: "
            "/10.250.19.102:54106 dest: /10.250.19.102:50010\n"
            "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
            "PacketResponder 1 for block blk_-1608999687919862906 terminating"
        ),
        source="hdfs-node-01",
        raw_text_format="hdfs",
        context={"environment": "lab"},
    )

    assert sequence.sequence_id == "blk_-1608999687919862906"
    assert sequence.source == "hdfs-node-01"
    assert len(sequence.events) == 2
    assert sequence.events[0].service == "hdfs"
    assert sequence.context["input_mode"] == "raw_log_text"
    assert sequence.context["raw_text_format"] == "hdfs"
    assert sequence.context["environment"] == "lab"


def test_normalize_prediction_request_accepts_raw_hdfs_request() -> None:
    request = make_raw_hdfs_request()
    sequence = normalize_prediction_request(request)

    assert sequence.sequence_id == "blk_-1608999687919862906"
    assert len(sequence.events) == 2
    assert sequence.events[0].message.startswith("Receiving block")
    assert sequence.events[1].message.startswith("PacketResponder 1 for block")


def test_normalize_raw_log_text_rejects_multiple_hdfs_block_ids() -> None:
    raw_text = (
        "081109 203615 148 INFO dfs.DataNode$DataXceiver: "
        "Receiving block blk_-111 src: /10.0.0.1:1 dest: /10.0.0.2:2\n"
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
        "PacketResponder 1 for block blk_-222 terminating"
    )

    try:
        normalize_raw_log_text(
            raw_text,
            source="hdfs-node-01",
            raw_text_format="hdfs",
        )
        assert False, "Expected ValueError for multiple HDFS block IDs"
    except ValueError as exc:
        assert "single HDFS block sequence" in str(exc)


def test_normalize_raw_log_text_generic_fallback() -> None:
    sequence = normalize_raw_log_text(
        "custom log line one\ncustom log line two",
        source="custom-source-01",
        raw_text_format="auto",
    )

    assert sequence.sequence_id.startswith("raw-generic-")
    assert len(sequence.events) == 2
    assert sequence.events[0].message == "custom log line one"
    assert sequence.events[1].message == "custom log line two"
    assert sequence.context["raw_text_format"] == "generic"