from __future__ import annotations

from fastapi import FastAPI

from app.core.errors import register_exception_handlers
from app.core.request_id import request_id_middleware
from app.core.size_limits import request_size_limit_middleware
from app.routes.model import router as model_router
from app.routes.predictions import router as predictions_router
from app.routes.system import router as system_router


# FastAPI application instance
# This is the main entry point of the RESTful API application.
# The application exposes the contract-defined endpoints for health checks,
# model metadata and prediction requests.
app = FastAPI(
    title="Secure RESTful API for ML-Based Anomaly Detection",
    version="1.0.0",
    summary="Contract-first API for anomaly detection in structured security telemetry.",
    description=(
        "This API provides machine learning-based anomaly detection for "
        "structured security telemetry. "
        "Version 1 supports the log_sequence / log_sequence_v1 runtime schema."
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
)


# Middleware registration
# Request-ID middleware assigns a unique server-generated request identifier to
# each request so that responses and errors can be traced consistently.
app.middleware("http")(request_id_middleware)

# Request-size middleware enforces contract-defined payload limits for the
# prediction endpoints.
app.middleware("http")(request_size_limit_middleware)


# Exception handler registration
# Centralized handlers ensure that validation errors, authentication failures
# and unexpected internal errors all return the standard API error schema.
register_exception_handlers(app)


# Router registration
# Routers are registered here so that endpoint definitions stay modular and
# separated from application startup configuration.
app.include_router(system_router)
app.include_router(model_router)
app.include_router(predictions_router)


# Temporary root endpoint
# This endpoint is not part of the final API contract.
# It is kept for early startup verification and can be removed later once the
# contract-defined routes are fully in place.
@app.get("/", tags=["startup"])
def root() -> dict[str, str]:
    return {
        "message": "API started successfully."
    }