"""`GET /api/demo-info`: what the public demo can do right now, for the main page.

Read-only: it reports the kill switches, whether each feature is configured, and the
daily caps with what is left today (from the proposal store, the voice quota and the
LLM call budget). It
never reserves a slot and never returns a secret, agent id or signed URL. Responses are
`Cache-Control: no-store` because the remaining counts change with every filed issue or
started voice session.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from slowapi import Limiter

from studiodesk.actions.github import GitHubIssues
from studiodesk.actions.proposals import ProposalStore
from studiodesk.api.deps import get_optional_github, get_settings
from studiodesk.config import Settings
from studiodesk.llm.limits import LLMCallBudget
from studiodesk.models.demo import DemoInfo, FeatureStatus, LLMStatus
from studiodesk.voice import VoiceSessionQuota

GITHUB_BASE_URL = "https://github.com"
NO_STORE = {"Cache-Control": "no-store"}


def sandbox_issues_url(repo: str | None) -> str | None:
    """Public Issues list of the sandbox repository (`owner/name`), or None if unset."""
    return f"{GITHUB_BASE_URL}/{repo}/issues" if repo else None


def actions_status(
    settings: Settings, github: GitHubIssues | None, proposals: ProposalStore | None
) -> FeatureStatus:
    """Issue filing: kill switch, GitHub configured, daily cap and slots left today."""
    cap = settings.actions_max_per_day
    remaining = proposals.remaining_today() if proposals is not None else 0
    return FeatureStatus(
        enabled=settings.actions_enabled,
        configured=github is not None,
        max_per_day=cap,
        remaining_today=cap if remaining is None else min(remaining, cap),
    )


def voice_status(
    settings: Settings, voice_configured: bool, quota: VoiceSessionQuota | None
) -> FeatureStatus:
    """Voice: kill switch, key and agent id configured, daily cap and sessions left today."""
    return FeatureStatus(
        enabled=settings.voice_enabled,
        configured=voice_configured,
        max_per_day=settings.voice_max_sessions_per_day,
        remaining_today=quota.remaining_today() if quota is not None else 0,
    )


def build_demo_info(
    settings: Settings,
    github: GitHubIssues | None,
    proposals: ProposalStore | None,
    voice_configured: bool,
    voice_quota: VoiceSessionQuota | None,
    llm_budget: LLMCallBudget | None = None,
) -> DemoInfo:
    """Assemble the public demo facts from settings and the startup collaborators."""
    repo = settings.github_repo
    return DemoInfo(
        source_repo_url=settings.source_repo_url,
        sandbox_repo=repo,
        sandbox_issues_url=sandbox_issues_url(repo),
        actions=actions_status(settings, github, proposals),
        voice=voice_status(settings, voice_configured, voice_quota),
        llm=LLMStatus(
            remaining_today=llm_budget.remaining_today()
            if llm_budget is not None
            else settings.llm_max_calls_per_day
        ),
    )


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Create the demo-info router, rate-limited per client IP by `limiter` at `rate_limit`."""
    router = APIRouter(tags=["demo"])

    @router.get("/api/demo-info", response_model=DemoInfo)
    @limiter.limit(rate_limit)
    def demo_info(
        request: Request,
        response: Response,
        settings: Annotated[Settings, Depends(get_settings)],
        github: Annotated[GitHubIssues | None, Depends(get_optional_github)],
    ) -> DemoInfo:
        """Return feature switches, caps and remaining counts (no secrets, never cached)."""
        response.headers.update(NO_STORE)
        state = request.app.state
        voice_configured = (
            getattr(state, "voice", None) is not None and settings.elevenlabs_agent_id is not None
        )
        return build_demo_info(
            settings,
            github,
            getattr(state, "proposals", None),
            voice_configured,
            getattr(state, "voice_quota", None),
            getattr(state, "llm_budget", None),
        )

    return router
