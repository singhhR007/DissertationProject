from __future__ import annotations

import os
from pathlib import Path


# Application configuration
# This module contains simple runtime configuration values for the current
# artefact stage. For now, configuration is loaded from environment variables
# with safe development defaults where appropriate.
API_VERSION = "1.0.0"

# NOTE:
# In a production deployment, this token must not be hard-coded and should be
# supplied securely via environment variables or a secret manager.
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "dev-secret-token")


# Model configuration
# Resolve paths relative to the project root:
# artefact/
#   app/
#     core/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ARTEFACT_PATH = Path(
    os.getenv(
        "MODEL_ARTEFACT_PATH",
        str(PROJECT_ROOT / "artefacts" / "models" / "hdfs_baseline_calibrated" / "hdfs_baseline.joblib"),
    )
)

DEPLOYED_MODEL_NAME = os.getenv(
    "DEPLOYED_MODEL_NAME",
    "baseline_log_anomaly_classifier",
)

DEPLOYED_MODEL_VERSION = os.getenv(
    "DEPLOYED_MODEL_VERSION",
    "v1.0.0",
)