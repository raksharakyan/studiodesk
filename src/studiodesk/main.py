"""FastAPI application factory and the module-level ASGI app used by uvicorn.

Building the app is cheap: the embedding model, the Qdrant client, the LLM client and the
outbound HTTP client are created in the lifespan startup (not in `create_app`), and only
when they were not injected. Resources created there are closed again on shutdown.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from studiodesk.actions.github import GitHubIssues
from studiodesk.actions.proposals import InMemoryProposalStore, ProposalStore
from studiodesk.actions.slack import SlackNotifier
from studiodesk.api import actions, ask, bugs, health, search
from studiodesk.api.deps import get_settings
from studiodesk.api.ratelimit import build_limiter, rate_limit_exceeded_handler
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder, SentenceTransformerEmbedder
from studiodesk.llm import AnthropicLLM, LLMClient
from studiodesk.logging import configure_logging
from studiodesk.middleware import BodySizeLimitMiddleware
from studiodesk.vectorstore import QdrantStore, build_qdrant_client

logger = logging.getLogger(__name__)

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


@dataclass(frozen=True)
class Injected:
    """Collaborators passed to `create_app`; None means "build from settings at startup"."""

    embedder: Embedder | None = None
    store: QdrantStore | None = None
    llm: LLMClient | None = None
    proposals: ProposalStore | None = None
    github: GitHubIssues | None = None
    slack: SlackNotifier | None = None


def _build_store(settings: Settings, stack: ExitStack) -> QdrantStore:
    """Create the Qdrant client (closed on shutdown) and wrap it in a store."""
    client = build_qdrant_client(settings)
    stack.callback(client.close)
    return QdrantStore(
        client, settings.qdrant_collection, upsert_batch_size=settings.qdrant_upsert_batch_size
    )


def _build_llm(settings: Settings, stack: ExitStack) -> LLMClient | None:
    """Create the Anthropic client if a key is configured, else None (routes return 503)."""
    if settings.anthropic_api_key is None:
        logger.warning("llm not configured: /ask and /bugs/check will return 503")
        return None
    llm = AnthropicLLM.from_settings(settings)
    stack.callback(llm.close)
    return llm


def _build_outbound(
    settings: Settings, injected: Injected, stack: ExitStack
) -> tuple[GitHubIssues | None, SlackNotifier]:
    """Create GitHub/Slack clients sharing one outbound httpx client (no redirects)."""
    http = httpx.Client(follow_redirects=False, timeout=settings.github_timeout_s)
    stack.callback(http.close)
    github = injected.github
    if github is None and settings.github_token is not None and settings.github_repo:
        github = GitHubIssues(
            http, settings.github_repo, settings.github_token, settings.github_timeout_s
        )
    if github is None:
        logger.warning("github issues not configured: no issues will be proposed")
    slack = injected.slack
    if slack is None:
        slack = SlackNotifier(http, settings.slack_webhook_url, settings.slack_timeout_s)
    return github, slack


def _make_lifespan(settings: Settings, injected: Injected) -> Lifespan:
    """Build the lifespan that puts every collaborator on `app.state`.

    Injected collaborators are used as-is; missing ones are created once at startup and
    the ones created here are closed on shutdown.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with ExitStack() as stack:
            app.state.store = (
                injected.store if injected.store is not None else _build_store(settings, stack)
            )
            app.state.embedder = (
                injected.embedder
                if injected.embedder is not None
                else await run_in_threadpool(SentenceTransformerEmbedder.from_settings, settings)
            )
            app.state.llm = (
                injected.llm if injected.llm is not None else _build_llm(settings, stack)
            )
            app.state.proposals = (
                injected.proposals
                if injected.proposals is not None
                else InMemoryProposalStore(
                    settings.action_ttl_s, max_pending=settings.action_max_pending
                )
            )
            app.state.github, app.state.slack = _build_outbound(settings, injected, stack)
            logger.info("resources ready", extra={"collection": app.state.store.collection})
            yield

    return lifespan


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
    llm: LLMClient | None = None,
    proposals: ProposalStore | None = None,
    github: GitHubIssues | None = None,
    slack: SlackNotifier | None = None,
) -> FastAPI:
    """Build the StudioDesk FastAPI app.

    Args:
        settings: Explicit settings to use. When given, they override the `get_settings`
            dependency for every route; when omitted, settings are read from the environment.
        embedder: Embedder for retrieval. Loaded from settings at startup if None.
        store: Vector store to search. Built from settings at startup if None.
        llm: LLM client. Built from settings at startup if None (None if no API key).
        proposals: Store for proposed actions. In-memory TTL store if None.
        github: GitHub issue client. Built from settings if None (None if not configured).
        slack: Slack notifier. Built from settings if None (no-op without a webhook).

    Returns:
        A configured FastAPI instance.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved)

    is_prod = resolved.app_env == "prod"
    injected = Injected(
        embedder=embedder,
        store=store,
        llm=llm,
        proposals=proposals,
        github=github,
        slack=slack,
    )
    app = FastAPI(
        title="StudioDesk",
        version=resolved.app_version,
        docs_url=None if is_prod else "/docs",
        redoc_url=None,
        openapi_url=None if is_prod else "/openapi.json",
        lifespan=_make_lifespan(resolved, injected),
    )
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings

    limiter = build_limiter(resolved.trusted_proxy_ips)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=resolved.max_request_bytes)
    app.include_router(health.router)
    app.include_router(search.build_router(limiter, resolved.search_rate_limit))
    app.include_router(ask.build_router(limiter, resolved.ask_rate_limit))
    app.include_router(bugs.build_router(limiter, resolved.bugs_rate_limit))
    app.include_router(actions.build_router(limiter, resolved.actions_rate_limit))

    logger.info("app created", extra={"env": resolved.app_env, "version": resolved.app_version})
    return app


app = create_app()
