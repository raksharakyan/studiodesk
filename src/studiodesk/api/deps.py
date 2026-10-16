"""FastAPI dependency providers.

Collaborators live on `app.state` (created or injected in the lifespan, see `main.py`).
A missing collaborator yields a generic 503, never a stack trace or configuration detail.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from studiodesk.actions.github import GitHubIssues
from studiodesk.actions.proposals import ProposalStore
from studiodesk.actions.slack import SlackNotifier
from studiodesk.agent.retrieval import Retriever
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.llm import LLMClient
from studiodesk.llm.limits import GatedLLM, LLMCallBudget, LLMConcurrencyGate
from studiodesk.vectorstore import QdrantStore
from studiodesk.voice import ElevenLabsClient, VoiceSessionQuota

SERVICE_UNAVAILABLE_DETAIL = "Search is temporarily unavailable"
LLM_UNAVAILABLE_DETAIL = "The assistant is temporarily unavailable"
LLM_NOT_CONFIGURED_DETAIL = "LLM not configured"
ACTIONS_UNAVAILABLE_DETAIL = "Actions are temporarily unavailable"
ISSUES_UNAVAILABLE_DETAIL = "Issue filing is temporarily unavailable"
VOICE_NOT_CONFIGURED_DETAIL = "Voice is not configured"
VOICE_DISABLED_DETAIL = "Voice is disabled"
VOICE_UNAVAILABLE_DETAIL = "Voice is temporarily unavailable"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings, built once from the environment.

    Tests and `create_app(settings=...)` override this via `app.dependency_overrides`.
    """
    return Settings()


def get_embedder(request: Request) -> Embedder:
    """Return the embedder loaded at startup, or 503 if it is not available."""
    embedder: Embedder | None = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL)
    return embedder


def get_store(request: Request) -> QdrantStore:
    """Return the vector store created at startup, or 503 if it is not available."""
    store: QdrantStore | None = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL)
    return store


def get_retriever(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    store: Annotated[QdrantStore, Depends(get_store)],
) -> Retriever:
    """Return a retriever over the startup embedder and store."""
    return Retriever(embedder, store)


def get_llm(request: Request) -> LLMClient:
    """Return the startup LLM client behind the app's concurrency gate (503 if none).

    `app.state.llm` stays the raw client; the shared gate and call budget live on
    `app.state.llm_gate` and `app.state.llm_budget`.
    """
    llm: LLMClient | None = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(status_code=503, detail=LLM_NOT_CONFIGURED_DETAIL)
    gate: LLMConcurrencyGate | None = getattr(request.app.state, "llm_gate", None)
    budget: LLMCallBudget | None = getattr(request.app.state, "llm_budget", None)
    return GatedLLM(llm, gate, budget) if gate is not None else llm


def get_proposal_store(request: Request) -> ProposalStore:
    """Return the proposal store created at startup, or 503 if it is not available."""
    proposals: ProposalStore | None = getattr(request.app.state, "proposals", None)
    if proposals is None:
        raise HTTPException(status_code=503, detail=ACTIONS_UNAVAILABLE_DETAIL)
    return proposals


def get_optional_github(request: Request) -> GitHubIssues | None:
    """Return the GitHub client, or None when issue filing is not configured."""
    github: GitHubIssues | None = getattr(request.app.state, "github", None)
    return github


def get_github(
    github: Annotated[GitHubIssues | None, Depends(get_optional_github)],
) -> GitHubIssues:
    """Return the GitHub client, or 503 when issue filing is not configured."""
    if github is None:
        raise HTTPException(status_code=503, detail=ISSUES_UNAVAILABLE_DETAIL)
    return github


def get_slack(request: Request) -> SlackNotifier:
    """Return the Slack notifier (a no-op one when no webhook is set), or 503 if missing."""
    slack: SlackNotifier | None = getattr(request.app.state, "slack", None)
    if slack is None:
        raise HTTPException(status_code=503, detail=ACTIONS_UNAVAILABLE_DETAIL)
    return slack


def require_voice_enabled(settings: Annotated[Settings, Depends(get_settings)]) -> None:
    """Raise 503 "Voice is disabled" when the VOICE_ENABLED kill switch is off."""
    if not settings.voice_enabled:
        raise HTTPException(
            status_code=503, detail=VOICE_DISABLED_DETAIL, headers={"Cache-Control": "no-store"}
        )


def get_voice_quota(request: Request) -> VoiceSessionQuota:
    """Return the daily voice session counter created at startup, or 503 if missing."""
    quota: VoiceSessionQuota | None = getattr(request.app.state, "voice_quota", None)
    if quota is None:
        raise HTTPException(status_code=503, detail=VOICE_UNAVAILABLE_DETAIL)
    return quota


def get_voice(request: Request) -> ElevenLabsClient:
    """Return the ElevenLabs client, or 503 when no ElevenLabs API key is configured."""
    voice: ElevenLabsClient | None = getattr(request.app.state, "voice", None)
    if voice is None:
        raise HTTPException(status_code=503, detail=VOICE_NOT_CONFIGURED_DETAIL)
    return voice


def get_voice_agent_id(settings: Annotated[Settings, Depends(get_settings)]) -> str:
    """Return the configured ElevenLabs agent id, or 503 when it is not set."""
    if not settings.elevenlabs_agent_id:
        raise HTTPException(status_code=503, detail=VOICE_NOT_CONFIGURED_DETAIL)
    return settings.elevenlabs_agent_id
