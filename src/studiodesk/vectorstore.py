"""Qdrant-backed vector store for chunks, plus the client factory and filter builder.

`QdrantStore` receives its `QdrantClient` from the caller, so tests can pass an in-memory
client (`QdrantClient(":memory:")`). Every client failure is re-raised as
`VectorStoreError` so callers can map it to a generic error without leaking details.
"""

import logging
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from studiodesk.config import Settings
from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import version_to_int
from studiodesk.models.search import SearchFilters

logger = logging.getLogger(__name__)

IN_MEMORY_LOCATION = ":memory:"
KEYWORD_INDEX_FIELDS = ("doc_type", "platform", "severity", "component", "version")
INTEGER_INDEX_FIELDS = ("version_num",)


class VectorStoreError(RuntimeError):
    """Any failure talking to the vector store. The message is safe to log, not to return."""


@contextmanager
def _store_errors(operation: str) -> Iterator[None]:
    """Re-raise any client exception as `VectorStoreError`, keeping the cause chained."""
    try:
        yield
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"qdrant {operation} failed: {type(exc).__name__}") from exc


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a client for Qdrant Cloud/server, or an embedded store when no URL is set.

    With `qdrant_url` set, connects over HTTP(S) with the API key and timeout from
    settings. Otherwise uses `qdrant_local_path` (`":memory:"` for a throwaway store).
    """
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=api_key.get_secret_value() if api_key is not None else None,
            timeout=settings.qdrant_timeout_s,
        )
    if settings.qdrant_local_path == IN_MEMORY_LOCATION:
        return QdrantClient(location=IN_MEMORY_LOCATION)
    return QdrantClient(path=settings.qdrant_local_path)


def build_filter(filters: SearchFilters | None) -> qm.Filter | None:
    """Translate validated search filters into a Qdrant `must` filter (None if empty).

    Lists become `MatchAny` conditions; the version range becomes a `Range` on the
    integer `version_num` payload field.
    """
    if filters is None:
        return None
    must: list[qm.Condition] = []
    keyword_filters: list[tuple[str, Sequence[str]]] = [
        ("doc_type", filters.doc_types),
        ("platform", filters.platforms),
        ("severity", filters.severities),
        ("component", filters.components),
    ]
    for key, values in keyword_filters:
        if values:
            unique = sorted({str(value) for value in values})
            must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=unique)))
    if filters.version_min or filters.version_max:
        must.append(
            qm.FieldCondition(
                key="version_num",
                range=qm.Range(
                    gte=version_to_int(filters.version_min) if filters.version_min else None,
                    lte=version_to_int(filters.version_max) if filters.version_max else None,
                ),
            )
        )
    return qm.Filter(must=must) if must else None


class ScoredChunk(BaseModel):
    """A stored chunk with its similarity score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: Chunk
    score: float


class QdrantStore:
    """Stores chunks as Qdrant points (cosine distance) and searches them with filters."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        """Use `client` for all calls against `collection`. The caller owns the client."""
        self._client = client
        self.collection = collection

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        """Create the collection and payload indexes if missing (or drop it first).

        Raises:
            VectorStoreError: on client errors, or if an existing collection has a
                different vector size than `dim`.
        """
        with _store_errors("ensure_collection"):
            exists = self._client.collection_exists(self.collection)
            if exists and recreate:
                self._client.delete_collection(self.collection)
                exists = False
            if exists:
                self._check_dimension(dim)
            else:
                self._client.create_collection(
                    self.collection,
                    vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
                )
            self._create_payload_indexes()

    def _check_dimension(self, dim: int) -> None:
        """Fail if the existing collection's vector size differs from `dim`."""
        vectors = self._client.get_collection(self.collection).config.params.vectors
        size = vectors.size if isinstance(vectors, qm.VectorParams) else None
        if size != dim:
            raise VectorStoreError(
                f"collection {self.collection!r} has vector size {size}, expected {dim}; "
                "re-run ingestion with --recreate"
            )

    def _create_payload_indexes(self) -> None:
        """Create keyword and integer payload indexes (idempotent on the server).

        Embedded (local/in-memory) Qdrant ignores indexes and warns; that warning is muted.
        """
        schemas = [(field, qm.PayloadSchemaType.KEYWORD) for field in KEYWORD_INDEX_FIELDS]
        schemas += [(field, qm.PayloadSchemaType.INTEGER) for field in INTEGER_INDEX_FIELDS]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in schemas:
                self._client.create_payload_index(
                    self.collection, field_name=field, field_schema=schema
                )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or overwrite one point per chunk; ids are deterministic per chunk.

        Raises:
            ValueError: if `chunks` and `vectors` differ in length.
            VectorStoreError: on client errors.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        points = [
            qm.PointStruct(
                id=chunk.point_id, vector=list(vector), payload=chunk.model_dump(mode="json")
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with _store_errors("upsert"):
            self._client.upsert(self.collection, points=points, wait=True)

    def search(
        self, vector: Sequence[float], filters: SearchFilters | None, top_k: int
    ) -> list[ScoredChunk]:
        """Return up to `top_k` chunks most similar to `vector` that match `filters`.

        Points whose payload no longer validates as a `Chunk` are skipped and logged.
        """
        with _store_errors("search"):
            response = self._client.query_points(
                self.collection,
                query=list(vector),
                query_filter=build_filter(filters),
                limit=top_k,
                with_payload=True,
            )
        results: list[ScoredChunk] = []
        for point in response.points:
            try:
                chunk = Chunk.model_validate(point.payload or {})
            except ValidationError:
                logger.warning("skipping point with invalid payload", extra={"point": point.id})
                continue
            results.append(ScoredChunk(chunk=chunk, score=point.score))
        return results

    def count(self) -> int:
        """Return the exact number of points in the collection."""
        with _store_errors("count"):
            return self._client.count(self.collection, exact=True).count
