"""Map internal failures to generic HTTP errors; details are logged, never returned."""

import logging

from fastapi import HTTPException

from studiodesk.actions.proposals import (
    ProposalCapacityError,
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


def llm_http_error(exc: LLMError) -> HTTPException:
    """503 if the provider is unavailable, else 502 (refusal, truncation, bad output)."""
    logger.warning("llm call failed", extra={"error": str(exc), "kind": type(exc).__name__})
    if isinstance(exc, LLMUnavailable):
        return HTTPException(status_code=503, detail=LLM_UNAVAILABLE_DETAIL)
    return HTTPException(status_code=502, detail=LLM_FAILED_DETAIL)


def vector_store_http_error() -> HTTPException:
    """Generic 503 for retrieval failures (call from an except block to log the cause)."""
    logger.exception("vector search failed")
    return HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL)


def proposal_http_error(exc: ProposalError) -> HTTPException:
    """404 unknown/wrong token, 409 used, 410 expired, 503 store full."""
    if isinstance(exc, ProposalUsedError):
        return HTTPException(status_code=409, detail=ACTION_USED_DETAIL)
    if isinstance(exc, ProposalExpiredError):
        return HTTPException(status_code=410, detail=ACTION_EXPIRED_DETAIL)
    if isinstance(exc, ProposalCapacityError):
        logger.warning("proposal store full")
        return HTTPException(status_code=503, detail=ACTIONS_UNAVAILABLE_DETAIL)
    return HTTPException(status_code=404, detail=ACTION_NOT_FOUND_DETAIL)
