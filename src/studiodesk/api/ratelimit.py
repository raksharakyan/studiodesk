"""Per-client rate limiting (slowapi) configured only from Settings."""

import os
import warnings

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

RATE_LIMITED_DETAIL = "Rate limit exceeded"


def build_limiter() -> Limiter:
    """Create an in-memory, per-client-IP limiter.

    slowapi would otherwise load a `.env` from the working directory; pointing it at the
    null device keeps every limit and toggle under `Settings` control.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Config file .* not found")
        return Limiter(
            key_func=get_remote_address,
            storage_uri="memory://",
            strategy="fixed-window",
            config_filename=os.devnull,
        )


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 429 with a Retry-After header when the limit window is known."""
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitExceeded) and exc.limit is not None:
        headers["Retry-After"] = str(exc.limit.limit.get_expiry())
    return JSONResponse({"detail": RATE_LIMITED_DETAIL}, status_code=429, headers=headers)
