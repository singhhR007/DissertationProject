from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import API_BEARER_TOKEN


# Bearer authentication scheme
# This security scheme is used for protected endpoints in version 1.
# auto_error=False lets us return our own controlled 401 response.
bearer_scheme = HTTPBearer(auto_error=False)


# Authentication dependency
def require_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    """
    Validate the bearer token for protected endpoints.

    The current implementation uses a single shared bearer token. This is
    sufficient for the artefact's current stage and matches the contract's
    allowance for a simple but consistently implemented token mechanism.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, API_BEARER_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )