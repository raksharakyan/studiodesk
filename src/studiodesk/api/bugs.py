"""`POST /bugs/check`: duplicate verdict, routing and (unless a duplicate) a proposed issue.

Nothing outward happens here. When the verdict is not `duplicate` and GitHub is
configured, the exact issue is stored server-side and returned as a preview with a
single-use `confirm_token`; filing needs `POST /actions/{action_id}/confirm`.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter

from studiodesk.actions.github import GitHubIssues
from studiodesk.actions.issue import build_issue_draft
from studiodesk.actions.proposals import ProposalError, ProposalStore
from studiodesk.agent.duplicates import check_duplicates
from studiodesk.agent.retrieval import Retriever
from studiodesk.agent.routing import route_bug
from studiodesk.api.deps import (
    get_llm,
    get_optional_github,
    get_proposal_store,
    get_retriever,
    get_settings,
)
from studiodesk.api.errors import (
    ACTIONS_DISABLED_DETAIL,
    DAILY_LIMIT_DETAIL,
    llm_http_error,
    proposal_http_error,
    vector_store_http_error,
)
from studiodesk.config import Settings
from studiodesk.llm import LLMClient, LLMError
from studiodesk.models.actions import ProposedAction
from studiodesk.models.bugs import (
    BugCheckResponse,
    DuplicateCheck,
    DuplicateVerdict,
    NewBugReport,
    RoutingResult,
)
from studiodesk.vectorstore import VectorStoreError

logger = logging.getLogger(__name__)


def propose_issue(
    report: NewBugReport,
    routing: RoutingResult,
    duplicates: DuplicateCheck,
    proposals: ProposalStore,
    repo: str,
    *,
    actions_enabled: bool,
) -> ProposedAction:
    """Return the issue preview, stored as a pending proposal when filing is possible.

    With filing disabled or today's cap reached, nothing is stored: the preview comes back
    without id/token/expiry and with `actions_disabled_reason`.
    """
    draft = build_issue_draft(report, routing, duplicates)
    reason = None
    if not actions_enabled:
        reason = ACTIONS_DISABLED_DETAIL
    elif proposals.daily_limit_reached():
        reason = DAILY_LIMIT_DETAIL
    if reason is not None:
        return ProposedAction(
            action_id=None,
            confirm_token=None,
            expires_at=None,
            repo=repo,
            preview=draft,
            actions_disabled_reason=reason,
        )
    proposal = proposals.create(draft)
    return ProposedAction(
        action_id=proposal.action_id,
        confirm_token=proposal.confirm_token,
        expires_at=proposal.expires_at,
        repo=repo,
        preview=draft,
    )


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Create the bugs router, rate-limited per client IP by `limiter` at `rate_limit`."""
    router = APIRouter(tags=["agent"])

    @router.post("/bugs/check", response_model=BugCheckResponse)
    @limiter.limit(rate_limit)
    def check_bug(
        request: Request,
        body: NewBugReport,
        retriever: Annotated[Retriever, Depends(get_retriever)],
        llm: Annotated[LLMClient, Depends(get_llm)],
        proposals: Annotated[ProposalStore, Depends(get_proposal_store)],
        github: Annotated[GitHubIssues | None, Depends(get_optional_github)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> BugCheckResponse:
        """Check `body` for duplicates, route it, and propose filing it unless duplicate."""
        try:
            routing = route_bug(body, retriever, k=settings.routing_k)
            duplicates: DuplicateCheck = check_duplicates(
                body,
                retriever,
                llm,
                search_k=settings.dup_search_k,
                candidate_threshold=settings.dup_candidate_threshold,
                auto_threshold=settings.dup_auto_threshold,
            )
        except VectorStoreError:
            raise vector_store_http_error() from None
        except LLMError as exc:
            raise llm_http_error(exc) from None
        proposed = None
        if duplicates.verdict is not DuplicateVerdict.DUPLICATE and github is not None:
            try:
                proposed = propose_issue(
                    body,
                    routing,
                    duplicates,
                    proposals,
                    github.repo,
                    actions_enabled=settings.actions_enabled,
                )
            except ProposalError as exc:
                raise proposal_http_error(exc) from None
        logger.info(
            "bug checked",
            extra={
                "verdict": duplicates.verdict.value,
                "candidates": len(duplicates.candidates),
                "proposed": proposed is not None,
            },
        )
        return BugCheckResponse(
            verdict=duplicates.verdict,
            duplicate_of=duplicates.duplicate_of,
            matched_report=duplicates.matched_report,
            candidates=duplicates.candidates,
            routing=routing,
            proposed_action=proposed,
        )

    return router
