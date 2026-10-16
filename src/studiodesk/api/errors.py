"""Map internal failures to generic HTTP errors; details are logged, never returned."""

import logging
import math
from collections.abc import Sequence

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from studiodesk.actions.proposals import (
    ProposalCapacityError,
    ProposalDailyLimitError,
    ProposalError,
    ProposalExpiredError,
    ProposalUsedError,
)
from studiodesk.api.deps import (
    ACTIONS_UNAVAILABLE_DETAIL,
    LLM_UNAVAILABLE_DETAIL,
    SERVICE_UNAVAILABLE_DETAIL,
)
from studiodesk.llm import LLMError, LLMUnavailable

logger = logging.getLogger(__name__)

LLM_FAILED_DETAIL = "The assistant could not complete this request"
ACTION_NOT_FOUND_DETAIL = "Action not found"
ACTION_USED_DETAIL = "Action was already confirmed or cancelled"
ACTION_EXPIRED_DETAIL = "Action expired"
ISSUE_FAILED_DETAIL = "The issue could not be created"
ACTIONS_DISABLED_DETAIL = "Issue filing is disabled"
DAILY_LIMIT_DETAIL = "Daily issue limit reached"


def llm_http_error(exc: LLMError) -> HTTPException:
    """503 if the provider is unavailable, else 502 (refusal, truncation, bad output).

    A provider-requested wait is passed on as a `Retry-After` header on the 503.
    """
    logger.warning("llm call failed", extra={"error": str(exc), "kind": type(exc).__name__})
    if isinstance(exc, LLMUnavailable):
        headers = (
            {"Retry-After": str(max(math.ceil(exc.retry_after_s), 1))}
            if exc.retry_after_s is not None
            else None
        )
        return HTTPException(status_code=503, detail=LLM_UNAVAILABLE_DETAIL, headers=headers)
    return HTTPException(status_code=502, detail=LLM_FAILED_DETAIL)


def vector_store_http_error() -> HTTPException:
    """Generic 503 for retrieval failures (call from an except block to log the cause)."""
    logger.exception("vector search failed")
    return HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL)


def proposal_http_error(exc: ProposalError) -> HTTPException:
    """404 unknown/wrong token, 409 used, 410 expired, 503 store full or daily cap."""
    if isinstance(exc, ProposalUsedError):
        return HTTPException(status_code=409, detail=ACTION_USED_DETAIL)
    if isinstance(exc, ProposalExpiredError):
        return HTTPException(status_code=410, detail=ACTION_EXPIRED_DETAIL)
    if isinstance(exc, ProposalDailyLimitError):
        logger.warning("daily issue limit reached")
        return HTTPException(status_code=503, detail=DAILY_LIMIT_DETAIL)
    if isinstance(exc, ProposalCapacityError):
        logger.warning("proposal store full")
        return HTTPException(status_code=503, detail=ACTIONS_UNAVAILABLE_DETAIL)
    return HTTPException(status_code=404, detail=ACTION_NOT_FOUND_DETAIL)


VALIDATION_ERROR_MSG = "Invalid value"
# Keys of unexpected fields are user-chosen; keep the echoed part of `loc` short.
MAX_LOC_PART_CHARS = 64


def _safe_loc(loc: Sequence[object]) -> list[str | int]:
    """Error location with ints kept and every other part as a truncated string."""
    return [part if isinstance(part, int) else str(part)[:MAX_LOC_PART_CHARS] for part in loc]


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """422 listing only each error's `loc`, `type` and a fixed `msg`.

    Pydantic's `input`, `ctx` and `msg` can echo request values back (M2 #8), so none of
    them is returned or logged.
    """
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    detail = [
        {
            "loc": _safe_loc(error.get("loc", ())),
            "type": str(error.get("type", "value_error")),
            "msg": VALIDATION_ERROR_MSG,
        }
        for error in errors
    ]
    return JSONResponse({"detail": detail}, status_code=422)
