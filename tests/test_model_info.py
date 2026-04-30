from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app


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


def test_model_info_requires_bearer_token() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/model/info")

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"


def test_model_info_returns_metadata(monkeypatch) -> None:
    client = TestClient(app)

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
        "app.routes.model.get_active_model_metadata",
        stub_get_active_model_metadata,
    )

    response = client.get(
        "/api/v1/model/info",
        headers={"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["model_name"] == "baseline_log_anomaly_classifier"
    assert body["model_version"] == "v1.0.0"
    assert body["primary_dataset"] == "HDFS"
    assert body["secondary_offline_benchmark"] == "OpenStack"
    assert body["score_type"] == "calibrated_anomalous_class_probability"
    assert body["threshold"] == 0.387672