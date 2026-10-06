"""FastAPI dependency providers."""

from functools import lru_cache

from studiodesk.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings, built once from the environment.

    Tests and `create_app(settings=...)` override this via `app.dependency_overrides`.
    """
    return Settings()
