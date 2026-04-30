from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_success() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()

    assert body["api_version"] == "1.0.0"
    assert "processed_at" in body