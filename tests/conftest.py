"""
Pytest configuration for the artefact test suite.

This file is automatically discovered by pytest and ensures that the project
root is on the Python import path so that the application package can be
imported by tests regardless of the current working directory.
It also fixes a deterministic bearer token for all protected-endpoint tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add the project root (the directory that contains the `app` package)
# to sys.path so tests can import `app.main`, `app.schemas`, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("API_BEARER_TOKEN", "test-token-for-pytest")