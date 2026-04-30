from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from app.schemas.prediction import PredictionRequest
from app.services.model_registry import get_active_model_artefact
from app.services.preprocessing import (
    NormalizedLogSequence,
    normalize_prediction_request,
)


PredictionLabel = Literal["normal", "anomalous"]


@dataclass(slots=True)
class InferenceResult:
    """
    Internal result object returned by the inference service.
    """

    prediction: PredictionLabel
    risk_score: float
    threshold: float
    model_version: str
    advisory: str | None


def _render_sequence_text(
    sequence: NormalizedLogSequence,
    *,
    mode: str,
) -> str:
    """
    Render a normalized sequence into the same text form used during training.
    """
    parts: list[str] = []

    for event in sequence.events:
        if mode == "messages":
            if event.message:
                parts.append(event.message)
            continue

        if mode == "enriched":
            event_parts: list[str] = []
            if event.component:
                event_parts.append(f"component={event.component}")
            if event.severity:
                event_parts.append(f"severity={event.severity}")
            if event.service:
                event_parts.append(f"service={event.service}")
            if event.message:
                event_parts.append(f"message={event.message}")

            if event_parts:
                parts.append(" ".join(event_parts))
            continue

        raise ValueError(f"Unsupported model text_mode: {mode}")

    return "\n".join(parts)


def _get_classifier_scores(classifier: Any, features: Any) -> np.ndarray:
    """
    Return one-dimensional raw classifier scores for calibration.
    """
    if hasattr(classifier, "decision_function"):
        return np.asarray(classifier.decision_function(features), dtype=float).reshape(-1)

    if hasattr(classifier, "predict_proba"):
        return np.asarray(classifier.predict_proba(features)[:, 1], dtype=float).reshape(-1)

    raise TypeError("Classifier must expose decision_function or predict_proba.")


def _predict_anomalous_probability(model_artefact: Any, features: Any) -> float:
    """
    Return the deployed anomalous-class probability for one feature row.
    """
    if model_artefact.calibrator is not None:
        raw_scores = _get_classifier_scores(model_artefact.classifier, features)
        calibrated = model_artefact.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        return float(calibrated[0])

    probabilities = model_artefact.classifier.predict_proba(features)[:, 1]
    return float(probabilities[0])


def _derive_prediction_label(risk_score: float, threshold: float) -> PredictionLabel:
    """
    Map a risk score to the public response label.
    """
    return "anomalous" if risk_score >= threshold else "normal"


def _derive_advisory(prediction: PredictionLabel) -> str:
    """
    Return a human-readable advisory message.
    """
    if prediction == "anomalous":
        return "Model output indicates anomalous behaviour. Human review is recommended."
    return "Model output indicates no anomalous behaviour at the configured threshold."


def predict_from_normalized_sequence(sequence: NormalizedLogSequence) -> InferenceResult:
    """
    Run inference for one normalized sequence using the deployed trained model.
    """
    model_artefact = get_active_model_artefact()
    metadata = model_artefact.metadata

    rendered_text = _render_sequence_text(sequence, mode=model_artefact.text_mode)
    transformed = model_artefact.vectorizer.transform([rendered_text])
    anomalous_probability = _predict_anomalous_probability(model_artefact, transformed)

    risk_score = round(anomalous_probability, 6)
    prediction = _derive_prediction_label(risk_score, metadata.threshold)

    return InferenceResult(
        prediction=prediction,
        risk_score=risk_score,
        threshold=round(metadata.threshold, 6),
        model_version=metadata.model_version,
        advisory=_derive_advisory(prediction),
    )


def predict_from_request(request: PredictionRequest) -> InferenceResult:
    """
    Normalize a validated API request and run inference on it.

    The request may provide either a structured log sequence or raw log text.
    Both input modes are normalized into the same internal sequence form before
    the model is executed.
    """
    normalized_sequence = normalize_prediction_request(request)
    return predict_from_normalized_sequence(normalized_sequence)