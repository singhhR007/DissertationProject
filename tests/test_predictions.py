from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app


@dataclass
class StubInferenceResult:
    prediction: str
    risk_score: float
    threshold: float
    model_version: str
    advisory: str | None


@dataclass
class StubModelMetadata:
    model_name: str
    model_version: str
    primary_dataset: str
    secondary_offline_benchmark: str
    telemetry_type: str
    telemetry_schema_version: str
    score_type: str
    threshold_selection_objective: str
    threshold: float
    last_updated: datetime


def make_prediction_payload() -> dict[str, object]:
    return {
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


def make_raw_text_prediction_payload() -> dict[str, object]:
    return {
        "timestamp": "2026-04-03T12:30:00Z",
        "source": "hdfs-node-01",
        "telemetry_type": "log_sequence",
        "telemetry_schema_version": "log_sequence_v1",
        "raw_log_text": (
            "081109 203615 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 src: "
            "/10.250.19.102:54106 dest: /10.250.19.102:50010\n"
            "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
            "PacketResponder 1 for block blk_-1608999687919862906 terminating"
        ),
        "raw_text_format": "hdfs",
        "context": {
            "environment": "lab",
        },
    }


def test_single_prediction_requires_bearer_token() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/predictions",
        json=make_prediction_payload(),
    )

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"


def test_single_prediction_returns_expected_response(monkeypatch) -> None:
    client = TestClient(app)

    def stub_predict_from_request(_payload):
        return StubInferenceResult(
            prediction="anomalous",
            risk_score=0.745647,
            threshold=0.387672,
            model_version="v1.0.0",
            advisory="Model output indicates anomalous behaviour. Human review is recommended.",
        )

    def stub_get_active_model_metadata():
        return StubModelMetadata(
            model_name="baseline_log_anomaly_classifier",
            model_version="v1.0.0",
            primary_dataset="HDFS",
            secondary_offline_benchmark="OpenStack",
            telemetry_type="log_sequence",
            telemetry_schema_version="log_sequence_v1",
            score_type="calibrated_anomalous_class_probability",
            threshold_selection_objective="max_f1_on_primary_validation_split",
            threshold=0.387672,
            last_updated=datetime(2026, 4, 9, 0, 0, 0, tzinfo=timezone.utc),
        )

    monkeypatch.setattr(
        "app.routes.predictions.predict_from_request",
        stub_predict_from_request,
    )
    monkeypatch.setattr(
        "app.routes.predictions.get_active_model_metadata",
        stub_get_active_model_metadata,
    )

    response = client.post(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"},
        json=make_prediction_payload(),
    )

    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == "anomalous"
    assert body["risk_score"] == 0.745647
    assert body["threshold"] == 0.387672
    assert body["score_type"] == "calibrated_anomalous_class_probability"
    assert body["model_version"] == "v1.0.0"
    assert "request_id" in body
    assert "processed_at" in body
    assert body["advisory"] is not None


def test_single_prediction_accepts_raw_log_text(monkeypatch) -> None:
    client = TestClient(app)

    def stub_predict_from_request(_payload):
        return StubInferenceResult(
            prediction="normal",
            risk_score=0.214563,
            threshold=0.387672,
            model_version="v1.0.0",
            advisory="Model output indicates no anomalous behaviour at the configured threshold.",
        )

    def stub_get_active_model_metadata():
        return StubModelMetadata(
            model_name="baseline_log_anomaly_classifier",
            model_version="v1.0.0",
            primary_dataset="HDFS",
            secondary_offline_benchmark="OpenStack",
            telemetry_type="log_sequence",
            telemetry_schema_version="log_sequence_v1",
            score_type="calibrated_anomalous_class_probability",
            threshold_selection_objective="max_f1_on_primary_validation_split",
            threshold=0.387672,
            last_updated=datetime(2026, 4, 9, 0, 0, 0, tzinfo=timezone.utc),
        )

    monkeypatch.setattr(
        "app.routes.predictions.predict_from_request",
        stub_predict_from_request,
    )
    monkeypatch.setattr(
        "app.routes.predictions.get_active_model_metadata",
        stub_get_active_model_metadata,
    )

    response = client.post(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"},
        json=make_raw_text_prediction_payload(),
    )

    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == "normal"
    assert body["risk_score"] == 0.214563
    assert body["threshold"] == 0.387672
    assert body["score_type"] == "calibrated_anomalous_class_probability"
    assert body["model_version"] == "v1.0.0"
    assert "request_id" in body
    assert "processed_at" in body
    assert body["advisory"] is not None


def test_single_prediction_rejects_both_log_sequence_and_raw_log_text() -> None:
    client = TestClient(app)

    payload = make_prediction_payload()
    payload["raw_log_text"] = (
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: "
        "PacketResponder 1 for block blk_-1608999687919862906 terminating"
    )
    payload["raw_text_format"] = "hdfs"

    response = client.post(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"},
        json=payload,
    )

    assert response.status_code == 422
    assert "Exactly one of log_sequence or raw_log_text must be provided." in response.text


def test_single_prediction_rejects_missing_both_input_modes() -> None:
    client = TestClient(app)

    payload = make_prediction_payload()
    payload.pop("log_sequence", None)

    response = client.post(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"},
        json=payload,
    )

    assert response.status_code == 422
    assert "Exactly one of log_sequence or raw_log_text must be provided." in response.text