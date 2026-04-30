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


def make_record(sequence_id: str) -> dict[str, object]:
    return {
        "timestamp": "2026-04-03T12:30:00Z",
        "source": "hdfs-node-01",
        "telemetry_type": "log_sequence",
        "telemetry_schema_version": "log_sequence_v1",
        "log_sequence": {
            "sequence_id": sequence_id,
            "events": [
                {
                    "message": f"Receiving block {sequence_id}",
                    "component": "DataNode",
                    "severity": "info",
                    "host": "10.250.19.102",
                    "service": "hdfs",
                }
            ],
        },
    }


def test_batch_prediction_returns_expected_response(monkeypatch) -> None:
    client = TestClient(app)

    calls = {"count": 0}

    def stub_predict_from_request(_payload):
        calls["count"] += 1
        if calls["count"] == 1:
            return StubInferenceResult(
                prediction="anomalous",
                risk_score=0.81,
                threshold=0.387672,
                model_version="v1.0.0",
                advisory="Model output indicates anomalous behaviour. Human review is recommended.",
            )
        return StubInferenceResult(
            prediction="normal",
            risk_score=0.08,
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

    payload = {
        "records": [
            make_record("blk_1"),
            make_record("blk_2"),
        ]
    }

    response = client.post(
        "/api/v1/predictions/batch",
        headers={"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"},
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()

    assert body["total_records"] == 2
    assert body["anomalous_records_detected"] == 1
    assert body["model_version"] == "v1.0.0"
    assert len(body["results"]) == 2

    assert body["results"][0]["record_index"] == 0
    assert body["results"][0]["prediction"] == "anomalous"
    assert body["results"][1]["record_index"] == 1
    assert body["results"][1]["prediction"] == "normal"