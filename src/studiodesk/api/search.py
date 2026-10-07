"""`POST /search`: semantic search over ingested chunks with metadata filters."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter

from studiodesk.api.deps import SERVICE_UNAVAILABLE_DETAIL, get_embedder, get_store
from studiodesk.embeddings import Embedder
from studiodesk.models.search import SearchHit, SearchRequest, SearchResponse
from studiodesk.vectorstore import QdrantStore, VectorStoreError

logger = logging.getLogger(__name__)


def run_search(body: SearchRequest, embedder: Embedder, store: QdrantStore) -> SearchResponse:
    """Embed the query, search with the validated filters and build the response.

    Raises:
        HTTPException: 503 with a generic message if the vector store fails; the details
            are logged, never returned.
    """
    vector = embedder.embed_query(body.query)
    try:
        results = store.search(vector, body.filters, body.top_k)
    except VectorStoreError:
        logger.exception("vector search failed")
        raise HTTPException(status_code=503, detail=SERVICE_UNAVAILABLE_DETAIL) from None
    return SearchResponse(hits=[SearchHit.from_chunk(r.chunk, r.score) for r in results])


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Create the search router, rate-limited per client IP by `limiter` at `rate_limit`."""
    router = APIRouter(tags=["search"])

    @router.post("/search", response_model=SearchResponse)
    @limiter.limit(rate_limit)
    def search(
        request: Request,
        body: SearchRequest,
        embedder: Annotated[Embedder, Depends(get_embedder)],
        store: Annotated[QdrantStore, Depends(get_store)],
    ) -> SearchResponse:
        """Return the chunks most similar to `query` that match all given filters."""
        return run_search(body, embedder, store)

    return router
