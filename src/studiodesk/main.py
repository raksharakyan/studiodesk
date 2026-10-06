"""FastAPI application factory and the module-level ASGI app used by uvicorn."""

import logging

from fastapi import FastAPI

from studiodesk.api import health
from studiodesk.api.deps import get_settings
from studiodesk.config import Settings
from studiodesk.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the StudioDesk FastAPI app.

    Args:
        settings: Explicit settings to use. When given, they override the `get_settings`
            dependency for every route; when omitted, settings are read from the environment.

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
    )
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    app.include_router(health.router)

    logger.info("app created", extra={"env": resolved.app_env, "version": resolved.app_version})
    return app


app = create_app()
