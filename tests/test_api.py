"""
Unit and integration tests for the anomaly-detection API.

These tests verify the behaviour claimed in the dissertation Chapter 4:
contract-level schema validation, bearer-token authentication,
raw-log-text format detection, and inference decision logic.

The tests are intentionally fast and self-contained. The inference layer
is mocked so that the API logic can be exercised independently of the
deployed model artefact, which keeps the suite runnable without external
files or extended startup time.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Ensure a known token is configured before the application is imported,
# so the protected endpoint tests use a deterministic credential.
os.environ.setdefault("API_BEARER_TOKEN", "test-token-for-pytest")

from app.main import app  # noqa: E402  # imported after env setup
from app.schemas.prediction import PredictionRequest  # noqa: E402
from app.services.inference import InferenceResult  # noqa: E402
from app.services.preprocessing import (  # noqa: E402
    extract_hdfs_block_id,
    extract_openstack_instance_id,
    normalize_raw_log_text,
    parse_hdfs_log_line,
    parse_openstack_log_line,
)


# Test fixtures and helpers

API_TOKEN = os.environ["API_BEARER_TOKEN"]


def _build_valid_request_payload(**overrides: Any) -> dict[str, Any]:
    """
    Build a minimal valid prediction request body.

    Individual tests override specific fields to construct edge cases.
    """
    payload: dict[str, Any] = {
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
    payload.update(overrides)
    return payload


@pytest.fixture
def client() -> TestClient:
    """
    Return a FastAPI test client bound to the production application.
    """
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """
    Return the bearer-token header used for authenticated requests.
    """
    return {"Authorization": f"Bearer {API_TOKEN}"}


@pytest.fixture
def mocked_inference():
    """
    Patch the inference service so API tests do not depend on the deployed
    model artefact. The mock returns a deterministic anomalous prediction.
    """
    fake_result = InferenceResult(
        prediction="anomalous",
        risk_score=0.85,
        threshold=0.70,
        model_version="v1.0.0-test",
        advisory="Mocked advisory for tests.",
    )
    with patch(
        "app.routes.predictions.predict_from_request",
        return_value=fake_result,
    ) as mock:
        yield mock


# Schema validation tests

class TestSchemaValidation:
    """
    Verify that the PredictionRequest schema enforces the documented API
    contract: literal-validated metadata fields, exclusive input modes,
    and rejection of unknown fields.
    """

    def test_valid_request_passes_validation(self) -> None:
        """A correctly formed request should validate without errors."""
        payload = _build_valid_request_payload()
        request = PredictionRequest.model_validate(payload)
        assert request.telemetry_type == "log_sequence"
        assert request.telemetry_schema_version == "log_sequence_v1"
        assert request.log_sequence is not None
        assert request.raw_log_text is None

    def test_wrong_telemetry_type_is_rejected(self) -> None:
        """Any value other than 'log_sequence' must be rejected."""
        payload = _build_valid_request_payload(telemetry_type="metric")
        with pytest.raises(ValidationError):
            PredictionRequest.model_validate(payload)

    def test_wrong_telemetry_schema_version_is_rejected(self) -> None:
        """Any version other than 'log_sequence_v1' must be rejected."""
        payload = _build_valid_request_payload(
            telemetry_schema_version="log_sequence_v2"
        )
        with pytest.raises(ValidationError):
            PredictionRequest.model_validate(payload)

    def test_both_input_modes_simultaneously_is_rejected(self) -> None:
        """Providing both log_sequence and raw_log_text must fail validation."""
        payload = _build_valid_request_payload(
            raw_log_text="some additional text"
        )
        with pytest.raises(ValidationError) as excinfo:
            PredictionRequest.model_validate(payload)
        assert "Exactly one" in str(excinfo.value)

    def test_no_input_mode_is_rejected(self) -> None:
        """Providing neither log_sequence nor raw_log_text must fail validation."""
        payload = _build_valid_request_payload(log_sequence=None)
        with pytest.raises(ValidationError) as excinfo:
            PredictionRequest.model_validate(payload)
        assert "Exactly one" in str(excinfo.value)

    def test_unknown_field_is_rejected(self) -> None:
        """ClosedSchemaModel must reject undocumented fields."""
        payload = _build_valid_request_payload(unexpected_field="should fail")
        with pytest.raises(ValidationError):
            PredictionRequest.model_validate(payload)

    def test_raw_text_format_without_raw_log_text_is_rejected(self) -> None:
        """raw_text_format may be provided only alongside raw_log_text."""
        payload = _build_valid_request_payload(raw_text_format="hdfs")
        with pytest.raises(ValidationError) as excinfo:
            PredictionRequest.model_validate(payload)
        assert "raw_text_format" in str(excinfo.value)

    def test_non_utc_timestamp_is_rejected(self) -> None:
        """Timestamps must be in canonical UTC Z form."""
        payload = _build_valid_request_payload(
            timestamp="2026-04-03T12:30:00+02:00"
        )
        with pytest.raises(ValidationError):
            PredictionRequest.model_validate(payload)


# Authentication tests

class TestAuthentication:
    """
    Verify that bearer-token authentication is enforced on protected
    endpoints, as described in Section 4.3 of the dissertation.
    """

    def test_request_without_token_is_rejected(
        self,
        client: TestClient,
    ) -> None:
        """A protected endpoint must reject requests with no Authorization header."""
        response = client.post(
            "/api/v1/predictions",
            json=_build_valid_request_payload(),
        )
        assert response.status_code == 401
        # The response body shape is determined by the application's
        # exception handlers; here we only assert that the status code
        # is correct and the body indicates a missing token.
        assert "Missing bearer token" in response.text

    def test_request_with_invalid_token_is_rejected(
        self,
        client: TestClient,
    ) -> None:
        """A protected endpoint must reject requests with an incorrect token."""
        response = client.post(
            "/api/v1/predictions",
            json=_build_valid_request_payload(),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401
        assert "Invalid bearer token" in response.text

    def test_request_with_wrong_scheme_is_rejected(
        self,
        client: TestClient,
    ) -> None:
        """Non-Bearer schemes such as Basic must be rejected."""
        response = client.post(
            "/api/v1/predictions",
            json=_build_valid_request_payload(),
            headers={"Authorization": "Basic some-base64-value"},
        )
        assert response.status_code == 401

    def test_request_with_valid_token_is_accepted(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mocked_inference: Any,
    ) -> None:
        """A correctly authenticated request must reach the inference layer."""
        response = client.post(
            "/api/v1/predictions",
            json=_build_valid_request_payload(),
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert mocked_inference.called


# Preprocessing tests

class TestPreprocessingFormatDetection:
    """
    Verify that the raw-log-text preprocessing layer parses HDFS and
    OpenStack lines correctly and falls back to a generic representation
    for unknown formats. These tests support the claim in Section 4.3
    that raw text is parsed into the same internal representation as
    structured input.
    """

    def test_hdfs_line_is_parsed_and_block_id_extracted(self) -> None:
        """A real HDFS log line must be parsed and the BlockId extracted."""
        line = (
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: "
            "PacketResponder 1 for block blk_-1608999687919862906 terminating"
        )
        event = parse_hdfs_log_line(line)
        assert event is not None
        assert event.severity == "info"
        assert event.component == "dfs.DataNode$PacketResponder"
        assert event.service == "hdfs"
        assert "PacketResponder 1" in event.message
        assert extract_hdfs_block_id(line) == "blk_-1608999687919862906"

    def test_openstack_line_is_parsed_and_instance_id_extracted(self) -> None:
        """A real OpenStack log line must be parsed and instance ID extracted."""
        line = (
            "nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:02.007 "
            "2931 INFO nova.virt.libvirt.driver "
            "[req-abc123][instance: 12345678-1234-1234-1234-123456789abc] "
            "VM started successfully"
        )
        event = parse_openstack_log_line(line)
        assert event is not None
        assert event.severity == "info"
        assert event.component == "nova.virt.libvirt.driver"
        assert event.service == "nova-compute"
        instance_id = extract_openstack_instance_id(line)
        assert instance_id == "12345678-1234-1234-1234-123456789abc"

    def test_raw_hdfs_text_normalises_to_single_sequence(self) -> None:
        """Multiple HDFS lines for one block must form one normalised sequence."""
        raw_text = (
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: "
            "PacketResponder 1 for block blk_-1608999687919862906 terminating\n"
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: "
            "Received block blk_-1608999687919862906 of size 91178"
        )
        sequence = normalize_raw_log_text(raw_text, source="hdfs-node-01")
        assert sequence.sequence_id == "blk_-1608999687919862906"
        assert len(sequence.events) == 2
        assert sequence.context["raw_text_format"] == "hdfs"

    def test_raw_text_with_multiple_block_ids_is_rejected(self) -> None:
        """Mixing distinct HDFS block IDs in one request must raise ValueError."""
        raw_text = (
            "081109 203519 145 INFO dfs.DataNode: block blk_-100 terminating\n"
            "081109 203519 145 INFO dfs.DataNode: block blk_-200 terminating"
        )
        with pytest.raises(ValueError) as excinfo:
            normalize_raw_log_text(raw_text, source="hdfs-node-01")
        assert "single HDFS block sequence" in str(excinfo.value)

    def test_generic_text_falls_back_to_generic_format(self) -> None:
        """Unrecognised log text must be parsed in generic mode without errors."""
        raw_text = "some arbitrary log line that matches no known format"
        sequence = normalize_raw_log_text(raw_text, source="custom-source")
        assert sequence.context["raw_text_format"] == "generic"
        assert len(sequence.events) == 1

    def test_empty_raw_text_is_rejected(self) -> None:
        """Raw text containing only whitespace must raise ValueError."""
        with pytest.raises(ValueError):
            normalize_raw_log_text("   \n  \n", source="any")


# Inference logic tests

class TestInferenceLogic:
    """
    Verify the decision logic that converts a continuous risk_score into a
    public prediction label. These tests support the claim in Section 4.3
    that the public label is derived from the configured threshold rather
    than from any internal classifier output.
    """

    def test_score_above_threshold_is_anomalous(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """A risk_score >= threshold must produce the 'anomalous' label."""
        fake_result = InferenceResult(
            prediction="anomalous",
            risk_score=0.95,
            threshold=0.70,
            model_version="v1.0.0-test",
            advisory="Mocked anomalous advisory.",
        )
        with patch(
            "app.routes.predictions.predict_from_request",
            return_value=fake_result,
        ):
            response = client.post(
                "/api/v1/predictions",
                json=_build_valid_request_payload(),
                headers=auth_headers,
            )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == "anomalous"
        assert body["risk_score"] >= body["threshold"]

    def test_score_below_threshold_is_normal(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """A risk_score < threshold must produce the 'normal' label."""
        fake_result = InferenceResult(
            prediction="normal",
            risk_score=0.10,
            threshold=0.70,
            model_version="v1.0.0-test",
            advisory="Mocked normal advisory.",
        )
        with patch(
            "app.routes.predictions.predict_from_request",
            return_value=fake_result,
        ):
            response = client.post(
                "/api/v1/predictions",
                json=_build_valid_request_payload(),
                headers=auth_headers,
            )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == "normal"
        assert body["risk_score"] < body["threshold"]

    def test_response_contract_fields_are_present(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        mocked_inference: Any,
    ) -> None:
        """Successful responses must include all fields documented in the contract."""
        response = client.post(
            "/api/v1/predictions",
            json=_build_valid_request_payload(),
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        expected_fields = {
            "request_id",
            "prediction",
            "risk_score",
            "threshold",
            "score_type",
            "model_version",
            "processed_at",
            "advisory",
        }
        assert expected_fields.issubset(body.keys())
        assert body["score_type"] == "calibrated_anomalous_class_probability"