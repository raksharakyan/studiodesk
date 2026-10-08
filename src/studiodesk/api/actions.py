"""`POST /actions/{action_id}/confirm` and `/cancel` for proposed outward actions.

Confirm executes the draft stored at proposal time; the request contributes only the
action id (path) and the single-use token (body). The token is consumed before GitHub is
called, so a failed or ambiguous creation is never retried with the same token (no
duplicate issues); the user re-submits through `/bugs/check` instead.

With `actions_enabled` false, confirm returns 503 "Issue filing is disabled" without
touching the token. `claim` reserves one of today's `actions_max_per_day` slots; it is
counted only when GitHub reports the issue created, and freed otherwise.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from slowapi import Limiter

from studiodesk.actions.github import GitHubError, GitHubIssues
from studiodesk.actions.proposals import ProposalError, ProposalStore
from studiodesk.actions.slack import SlackNotifier
from studiodesk.api.deps import get_github, get_proposal_store, get_settings, get_slack
from studiodesk.api.errors import ACTIONS_DISABLED_DETAIL, ISSUE_FAILED_DETAIL, proposal_http_error
from studiodesk.config import Settings
from studiodesk.models.actions import (
    ACTION_ID_PATTERN,
    CancelResponse,
    ConfirmRequest,
    ConfirmResponse,
)

logger = logging.getLogger(__name__)

ActionIdPath = Annotated[str, Path(pattern=ACTION_ID_PATTERN)]


def require_actions_enabled(settings: Annotated[Settings, Depends(get_settings)]) -> None:
    """503 "Issue filing is disabled" before any token is looked at or consumed."""
    if not settings.actions_enabled:
        raise HTTPException(status_code=503, detail=ACTIONS_DISABLED_DETAIL)


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Create the actions router, rate-limited per client IP by `limiter` at `rate_limit`."""
    router = APIRouter(tags=["actions"])

    @router.post(
        "/actions/{action_id}/confirm",
        response_model=ConfirmResponse,
        dependencies=[Depends(require_actions_enabled)],
    )
    @limiter.limit(rate_limit)
    def confirm_action(
        request: Request,
        action_id: ActionIdPath,
        body: ConfirmRequest,
        proposals: Annotated[ProposalStore, Depends(get_proposal_store)],
        github: Annotated[GitHubIssues, Depends(get_github)],
        slack: Annotated[SlackNotifier, Depends(get_slack)],
    ) -> ConfirmResponse:
        """File the stored issue, then send the Slack alert (or skip it if unconfigured)."""
        try:
            draft = proposals.claim(action_id, body.confirm_token)
        except ProposalError as exc:
            raise proposal_http_error(exc) from None
        try:
            issue = github.create(draft)
        except GitHubError as exc:
            proposals.release(action_id, created=False)
            logger.warning("issue creation failed", extra={"error": str(exc)})
            raise HTTPException(status_code=502, detail=ISSUE_FAILED_DETAIL) from None
        except BaseException:
            proposals.release(action_id, created=False)
            raise
        proposals.release(action_id, created=True)
        logger.info("issue filed", extra={"issue_number": issue.number})
        slack_status = slack.notify_issue_created(issue, draft)
        return ConfirmResponse(issue_number=issue.number, issue_url=issue.url, slack=slack_status)

    @router.post("/actions/{action_id}/cancel", response_model=CancelResponse)
    @limiter.limit(rate_limit)
    def cancel_action(
        request: Request,
        action_id: ActionIdPath,
        body: ConfirmRequest,
        proposals: Annotated[ProposalStore, Depends(get_proposal_store)],
    ) -> CancelResponse:
        """Cancel a pending proposal so its token can no longer be confirmed."""
        try:
            proposals.cancel(action_id, body.confirm_token)
        except ProposalError as exc:
            raise proposal_http_error(exc) from None
        return CancelResponse()

    return router
