"""FastAPI dependency providers."""

from functools import lru_cache

from fastapi import HTTPException, Request

from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.vectorstore import QdrantStore

SERVICE_UNAVAILABLE_DETAIL = "Search is temporarily unavailable"


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
