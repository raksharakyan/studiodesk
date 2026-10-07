"""FastAPI application factory and the module-level ASGI app used by uvicorn.

Building the app is cheap: the embedding model and the Qdrant client are created in the
lifespan startup (not in `create_app`), and only when they were not injected.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from studiodesk.api import health, search
from studiodesk.api.deps import get_settings
from studiodesk.api.ratelimit import build_limiter, rate_limit_exceeded_handler
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder, SentenceTransformerEmbedder
from studiodesk.logging import configure_logging
from studiodesk.middleware import BodySizeLimitMiddleware
from studiodesk.vectorstore import QdrantStore, build_qdrant_client

logger = logging.getLogger(__name__)

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def _make_lifespan(
    settings: Settings, embedder: Embedder | None, store: QdrantStore | None
) -> Lifespan:
    """Build the lifespan that provides `app.state.embedder` and `app.state.store`.

    Injected collaborators are used as-is; missing ones are created once at startup. A
    Qdrant client created here is closed again on shutdown.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_client = None
        if store is None:
            owned_client = build_qdrant_client(settings)
            app.state.store = QdrantStore(owned_client, settings.qdrant_collection)
        else:
            app.state.store = store
        try:
            app.state.embedder = (
                embedder
                if embedder is not None
                else await run_in_threadpool(SentenceTransformerEmbedder.from_settings, settings)
            )
            logger.info("search resources ready", extra={"collection": app.state.store.collection})
            yield
        finally:
            if owned_client is not None:
                owned_client.close()

    return lifespan


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
) -> FastAPI:
    """Build the StudioDesk FastAPI app.

    Args:
        settings: Explicit settings to use. When given, they override the `get_settings`
            dependency for every route; when omitted, settings are read from the environment.
        embedder: Embedder to serve `/search` with. Loaded from settings at startup if None.
        store: Vector store to search. Built from settings at startup if None.

    Returns:
        A configured FastAPI instance.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved)

    is_prod = resolved.app_env == "prod"
    app = FastAPI(
        title="StudioDesk",
        version=resolved.app_version,
        docs_url=None if is_prod else "/docs",
        redoc_url=None,
        openapi_url=None if is_prod else "/openapi.json",
        lifespan=_make_lifespan(resolved, embedder, store),
    )
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings

    limiter = build_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=resolved.max_request_bytes)
    app.include_router(health.router)
    app.include_router(search.build_router(limiter, resolved.search_rate_limit))

    logger.info("app created", extra={"env": resolved.app_env, "version": resolved.app_version})
    return app


app = create_app()
