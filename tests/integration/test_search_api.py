"""POST /search end to end: fake embedder + in-memory Qdrant injected into the app."""

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.models.search import SNIPPET_MAX_CHARS, SearchFilters
from studiodesk.vectorstore import QdrantStore, ScoredChunk, VectorStoreError

GENERIC_503 = {"detail": "Search is temporarily unavailable"}


@pytest.fixture
def search_client(
    settings: Settings, fake_embedder: Embedder, ingested_store: QdrantStore
) -> Iterator[TestClient]:
    with TestClient(create_app(settings, embedder=fake_embedder, store=ingested_store)) as c:
        yield c


# --- happy path --------------------------------------------------------------------------


def test_search_returns_hits_with_metadata(search_client: TestClient) -> None:
    response = search_client.post("/search", json={"query": "save corrupted after cryo sleep"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    hits = response.json()["hits"]
    assert len(hits) == 5
    assert set(hits[0]) == {"score", "doc_id", "doc_type", "title", "snippet", "metadata"}
    assert set(hits[0]["metadata"]) == {
        "chunk_index",
        "platform",
        "version",
        "severity",
        "component",
        "created_at",
        "duplicate_of",
        "related_bug",
    }
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_and_snippet_bound(search_client: TestClient) -> None:
    response = search_client.post("/search", json={"query": "crash stack trace", "top_k": 20})

    hits = response.json()["hits"]
    assert len(hits) == 20
    assert all(len(h["snippet"]) <= SNIPPET_MAX_CHARS for h in hits)


@pytest.mark.parametrize(
    ("filters", "check"),
    [
        ({"platforms": ["switch"]}, lambda h: "switch" in h["metadata"]["platform"]),
        ({"severities": ["critical"]}, lambda h: h["metadata"]["severity"] == "critical"),
        ({"doc_types": ["patch_note"]}, lambda h: h["doc_type"] == "patch_note"),
        ({"components": ["netcode"]}, lambda h: h["metadata"]["component"] == "netcode"),
        (
            {"version_min": "1.1.0", "version_max": "1.1.9"},
            lambda h: h["metadata"]["version"].startswith("1.1."),
        ),
    ],
)
def test_search_filters_applied_end_to_end(
    search_client: TestClient, filters: dict[str, Any], check: Any
) -> None:
    response = search_client.post(
        "/search", json={"query": "game broken", "top_k": 20, "filters": filters}
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits
    assert all(check(h) for h in hits)


def test_search_with_no_matches_returns_empty_list(search_client: TestClient) -> None:
    body = {"query": "x", "filters": {"doc_types": ["doc"], "severities": ["low"]}}

    response = search_client.post("/search", json=body)

    assert response.status_code == 200
    assert response.json() == {"hits": []}


# --- request validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 21},
        {"query": "ok", "filters": {"platforms": ["n64"]}},
        {"query": "ok", "extra": True},
        {"query": "ok", "filters": {"must": [{"key": "doc_id", "match": {"value": "x"}}]}},
        {"query": "ok", "filters": {"version_min": "2.0.0", "version_max": "1.0.0"}},
        {"query": ["a", "b"]},
        {},
    ],
)
def test_invalid_body_is_422(search_client: TestClient, body: dict[str, Any]) -> None:
    assert search_client.post("/search", json=body).status_code == 422


def test_non_json_body_is_422(search_client: TestClient) -> None:
    response = search_client.post(
        "/search", content=b"query=hello", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422


def test_get_not_allowed(search_client: TestClient) -> None:
    assert search_client.get("/search").status_code == 405


# --- body size limit (413) ---------------------------------------------------------------


@pytest.fixture
def small_body_client(fake_embedder: Embedder, ingested_store: QdrantStore) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, app_env="test", max_request_bytes=200)
    with TestClient(create_app(settings, embedder=fake_embedder, store=ingested_store)) as c:
        yield c


def test_oversized_body_with_content_length_is_413(small_body_client: TestClient) -> None:
    response = small_body_client.post("/search", json={"query": "x" * 400})

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


def test_oversized_streamed_body_is_413(small_body_client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"query": "'
        for _ in range(50):
            yield b"x" * 20
        yield b'"}'

    response = small_body_client.post(
        "/search", content=chunks(), headers={"content-type": "application/json"}
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


def test_small_streamed_body_is_accepted(small_body_client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"query": '
        yield b'"cryo save"}'

    response = small_body_client.post(
        "/search", content=chunks(), headers={"content-type": "application/json"}
    )

    assert response.status_code == 200


def test_body_at_limit_is_accepted(small_body_client: TestClient) -> None:
    body = b'{"query": "' + b"a" * (200 - 13) + b'"}'
    assert len(body) == 200

    response = small_body_client.post(
        "/search", content=body, headers={"content-type": "application/json"}
    )

    assert response.status_code == 200


@pytest.mark.parametrize("value", ["abc", "-1", "1e3"])
def test_malformed_content_length_is_400(small_body_client: TestClient, value: str) -> None:
    response = small_body_client.post(
        "/search",
        content=b'{"query": "x"}',
        headers={"content-type": "application/json", "content-length": value},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Content-Length header"}


def test_body_limit_applies_to_other_routes(small_body_client: TestClient) -> None:
    assert small_body_client.post("/health", content=b"x" * 500).status_code == 413


def test_middleware_rejects_non_positive_limit() -> None:
    from studiodesk.middleware import BodySizeLimitMiddleware

    with pytest.raises(ValueError, match="positive"):
        BodySizeLimitMiddleware(app=lambda *a: None, max_bytes=0)  # type: ignore[arg-type]


# --- rate limit (429) --------------------------------------------------------------------


def test_rate_limit_returns_generic_429(
    fake_embedder: Embedder, ingested_store: QdrantStore
) -> None:
    settings = Settings(_env_file=None, app_env="test", search_rate_limit="3/minute")
    with TestClient(create_app(settings, embedder=fake_embedder, store=ingested_store)) as c:
        codes = [c.post("/search", json={"query": "save"}).status_code for _ in range(4)]
        limited = c.post("/search", json={"query": "save"})
        health = c.get("/health")

    assert codes == [200, 200, 200, 429]
    assert limited.status_code == 429
    assert limited.json() == {"detail": "Rate limit exceeded"}
    assert int(limited.headers["retry-after"]) > 0
    assert health.status_code == 200, "rate limit only applies to /search"


def test_rate_limit_state_is_per_app(fake_embedder: Embedder, ingested_store: QdrantStore) -> None:
    settings = Settings(_env_file=None, app_env="test", search_rate_limit="1/minute")
    for _ in range(2):
        with TestClient(create_app(settings, embedder=fake_embedder, store=ingested_store)) as c:
            assert c.post("/search", json={"query": "a"}).status_code == 200
            assert c.post("/search", json={"query": "a"}).status_code == 429


@pytest.mark.parametrize("limit", ["0/minute", "10/fortnight", "lots", "10 per minute", ""])
def test_invalid_rate_limit_setting_rejected(limit: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None, search_rate_limit=limit)


# --- store failures: generic 503, details only in (redacted) logs ------------------------

LEAKY = "qk-LEAKY-API-KEY-123"


class _FailingStore(QdrantStore):
    def __init__(self) -> None:
        self.collection = "failing"

    def search(
        self, vector: Sequence[float], filters: SearchFilters | None, top_k: int
    ) -> list[ScoredChunk]:
        try:
            raise ConnectionError(f"connect to https://u:{LEAKY}@cluster.internal:6333 refused")
        except ConnectionError as exc:
            raise VectorStoreError("qdrant search failed: ConnectionError") from exc


def test_store_failure_is_generic_503_without_details(
    fake_embedder: Embedder, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(_env_file=None, app_env="test", qdrant_api_key=SecretStr(LEAKY))
    with TestClient(create_app(settings, embedder=fake_embedder, store=_FailingStore())) as c:
        response = c.post("/search", json={"query": "anything"})

    assert response.status_code == 503
    assert response.json() == GENERIC_503
    assert "ConnectionError" not in response.text
    assert "cluster.internal" not in response.text
    out = capsys.readouterr().out
    assert "vector search failed" in out
    assert LEAKY not in out


def test_missing_collection_is_generic_503(client: TestClient) -> None:
    response = client.post("/search", json={"query": "anything"})

    assert response.status_code == 503
    assert response.json() == GENERIC_503
    assert "qdrant" not in response.text.lower()


def test_search_before_lifespan_is_503(settings: Settings) -> None:
    app = create_app(settings, embedder=None, store=None)
    c = TestClient(app)  # no context manager: lifespan never runs, state is empty

    response = c.post("/search", json={"query": "x"})

    assert response.status_code == 503
    assert response.json() == GENERIC_503


def test_lifespan_builds_local_store_when_not_injected(fake_embedder: Embedder) -> None:
    settings = Settings(_env_file=None, app_env="test", qdrant_local_path=":memory:")
    app = create_app(settings, embedder=fake_embedder)

    with TestClient(app) as c:
        assert isinstance(app.state.store, QdrantStore)
        assert app.state.embedder is fake_embedder
        assert c.post("/search", json={"query": "x"}).json() == GENERIC_503
