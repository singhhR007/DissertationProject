from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib

from app.core.config import (
    DEPLOYED_MODEL_NAME,
    DEPLOYED_MODEL_VERSION,
    MODEL_ARTEFACT_PATH,
)


@dataclass(frozen=True, slots=True)
class DeployedModelMetadata:
    """
    Central metadata object describing the currently active deployed model.
    """

    model_name: str
    model_version: str
    primary_dataset: Literal["HDFS"]
    secondary_offline_benchmark: Literal["OpenStack"]
    telemetry_type: Literal["log_sequence"]
    telemetry_schema_version: Literal["log_sequence_v1"]
    score_type: Literal["calibrated_anomalous_class_probability"]
    threshold_selection_objective: Literal["max_f1_on_primary_validation_split"]
    threshold: float
    last_updated: datetime


@dataclass(frozen=True, slots=True)
class LoadedModelArtefact:
    """
    In-memory representation of the deployed trained model artefact.
    """

    metadata: DeployedModelMetadata
    model_type: str
    positive_label: str
    negative_label: str
    text_mode: str
    score_type: str
    vectorizer: Any
    classifier: Any
    calibrator: Any | None
    calibration_method: str | None
    calibration_input: str | None
    feature_config: dict[str, Any]
    source_path: Path


def _read_last_updated_from_path(path: Path) -> datetime:
    """
    Derive the last-updated timestamp from the artefact file modification time.
    """
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    """
    Validate that a loaded object field is a mapping with string keys.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} must contain only string keys.")
        normalized[key] = item

    return normalized


def _load_model_artefact_from_disk(path: Path) -> LoadedModelArtefact:
    """
    Load the trained model artefact from disk and validate the expected shape.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found: {path}")

    loaded = joblib.load(path)
    raw = _require_mapping(loaded, field_name="model_artefact")

    required_keys = {
        "model_type",
        "positive_label",
        "negative_label",
        "text_mode",
        "score_type",
        "threshold",
        "vectorizer",
        "classifier",
        "feature_config",
    }
    missing_keys = sorted(required_keys.difference(raw.keys()))
    if missing_keys:
        raise KeyError(
            f"Model artefact is missing required keys: {', '.join(missing_keys)}"
        )

    score_type = str(raw["score_type"])
    if score_type != "calibrated_anomalous_class_probability":
        raise ValueError(
            "Loaded model artefact does not expose the expected calibrated score_type."
        )

    feature_config = dict(
        _require_mapping(raw["feature_config"], field_name="feature_config")
    )

    threshold = float(raw["threshold"])
    metadata = DeployedModelMetadata(
        model_name=DEPLOYED_MODEL_NAME,
        model_version=DEPLOYED_MODEL_VERSION,
        primary_dataset="HDFS",
        secondary_offline_benchmark="OpenStack",
        telemetry_type="log_sequence",
        telemetry_schema_version="log_sequence_v1",
        score_type="calibrated_anomalous_class_probability",
        threshold_selection_objective="max_f1_on_primary_validation_split",
        threshold=threshold,
        last_updated=_read_last_updated_from_path(path),
    )

    return LoadedModelArtefact(
        metadata=metadata,
        model_type=str(raw["model_type"]),
        positive_label=str(raw["positive_label"]),
        negative_label=str(raw["negative_label"]),
        text_mode=str(raw["text_mode"]),
        score_type=score_type,
        vectorizer=raw["vectorizer"],
        classifier=raw["classifier"],
        calibrator=raw.get("calibrator"),
        calibration_method=(
            str(raw["calibration_method"])
            if raw.get("calibration_method") is not None
            else None
        ),
        calibration_input=(
            str(raw["calibration_input"])
            if raw.get("calibration_input") is not None
            else None
        ),
        feature_config=feature_config,
        source_path=path,
    )


@lru_cache(maxsize=1)
def get_active_model_artefact() -> LoadedModelArtefact:
    """
    Return the currently active loaded model artefact.
    """
    return _load_model_artefact_from_disk(MODEL_ARTEFACT_PATH)


def get_active_model_metadata() -> DeployedModelMetadata:
    """
    Return metadata for the currently active deployed model.
    """
    return get_active_model_artefact().metadata


def is_model_ready() -> bool:
    """
    Return whether the active model is available for inference.
    """
    try:
        get_active_model_artefact()
    except (FileNotFoundError, TypeError, KeyError, OSError, ValueError):
        return False
    return True