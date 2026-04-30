from __future__ import annotations

from app.schemas.error import ErrorResponse


# Reusable OpenAPI response documentation
# These dictionaries are used only for documentation so that the documented
# OpenAPI responses match the runtime behaviour of the centralized exception
# handlers.


UNAUTHORIZED_RESPONSE = {
    401: {
        "model": ErrorResponse,
        "description": "Unauthorized. Missing or invalid bearer token.",
    }
}

FORBIDDEN_RESPONSE = {
    403: {
        "model": ErrorResponse,
        "description": "Forbidden. Authenticated caller lacks permission.",
    }
}

VALIDATION_ERROR_RESPONSE = {
    422: {
        "model": ErrorResponse,
        "description": "Request payload failed schema validation.",
    }
}

PAYLOAD_TOO_LARGE_RESPONSE = {
    413: {
        "model": ErrorResponse,
        "description": "Request body exceeds the configured maximum size.",
    }
}

UNSUPPORTED_MEDIA_TYPE_RESPONSE = {
    415: {
        "model": ErrorResponse,
        "description": "Unsupported media type. JSON is required.",
    }
}

INTERNAL_ERROR_RESPONSE = {
    500: {
        "model": ErrorResponse,
        "description": "Internal server error.",
    }
}

SERVICE_UNAVAILABLE_RESPONSE = {
    503: {
        "model": ErrorResponse,
        "description": "Service unavailable.",
    }
}


# Combined response sets

PROTECTED_GET_ERROR_RESPONSES = {
    **UNAUTHORIZED_RESPONSE,
    **FORBIDDEN_RESPONSE,
    **INTERNAL_ERROR_RESPONSE,
    **SERVICE_UNAVAILABLE_RESPONSE,
}

PREDICTION_POST_ERROR_RESPONSES = {
    **UNAUTHORIZED_RESPONSE,
    **FORBIDDEN_RESPONSE,
    **PAYLOAD_TOO_LARGE_RESPONSE,
    **UNSUPPORTED_MEDIA_TYPE_RESPONSE,
    **VALIDATION_ERROR_RESPONSE,
    **INTERNAL_ERROR_RESPONSE,
    **SERVICE_UNAVAILABLE_RESPONSE,
}