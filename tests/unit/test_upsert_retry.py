"""QdrantStore upsert batching and retry with a stub client and an injected sleep."""

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from studiodesk.config import Settings
from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import DocType
from studiodesk.vectorstore import QdrantStore, VectorStoreError, is_transient

LEAKY_URL = "https://admin:qk-SECRET-KEY@cluster.internal.example:6333/collections/x"


def _unexpected(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status, "reason", f"body {LEAKY_URL}".encode(), httpx.Headers())


def _read_timeout() -> ResponseHandlingException:
    return ResponseHandlingException(httpx.ReadTimeout(f"timed out talking to {LEAKY_URL}"))


class StubClient:
    """Records upsert batch sizes; raises queued errors on successive upsert calls."""

    def __init__(self, failures: list[BaseException] | None = None) -> None:
        self.failures = list(failures or [])
        self.attempts: list[int] = []
        self.stored: list[int] = []

    def upsert(self, collection: str, *, points: list[qm.PointStruct], wait: bool) -> None:
        assert wait is True
        self.attempts.append(len(points))
        if self.failures:
            raise self.failures.pop(0)
        self.stored.append(len(points))


def _chunks(n: int) -> tuple[list[Chunk], list[list[float]]]:
    chunks = [
        Chunk(doc_id=f"D{i}", doc_type=DocType.DOC, chunk_index=0, title="t", text="x")
        for i in range(n)
    ]
    return chunks, [[1.0, 0.0]] * n


def _store(client: StubClient, sleeps: list[float], **kwargs: Any) -> QdrantStore:
    return QdrantStore(client, "c", sleep=sleeps.append, **kwargs)  # type: ignore[arg-type]


# --- settings ----------------------------------------------------------------------------


def test_defaults() -> None:
    s = Settings(_env_file=None)

    assert s.qdrant_timeout_s == 30
    assert s.qdrant_upsert_batch_size == 32


@pytest.mark.parametrize("value", [0, -1, 1025])
def test_upsert_batch_size_bounds(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_upsert_batch_size=value)


@pytest.mark.parametrize("value", [1, 1024])
def test_upsert_batch_size_edges_accepted(value: int) -> None:
    assert (
        Settings(_env_file=None, qdrant_upsert_batch_size=value).qdrant_upsert_batch_size == value
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upsert_batch_size": 0},
        {"upsert_batch_size": -1},
        {"upsert_attempts": 0},
        {"retry_backoff_s": -0.1},
    ],
)
def test_constructor_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        QdrantStore(StubClient(), "c", **kwargs)  # type: ignore[arg-type]


def test_zero_backoff_and_single_attempt_allowed() -> None:
    QdrantStore(StubClient(), "c", upsert_attempts=1, retry_backoff_s=0)  # type: ignore[arg-type]


# --- batching ----------------------------------------------------------------------------


def test_points_sent_in_batches_of_32() -> None:
    client, sleeps = StubClient(), []
    _store(client, sleeps).upsert(*_chunks(70))

    assert client.attempts == [32, 32, 6]
    assert client.stored == [32, 32, 6]
    assert sleeps == []


def test_custom_batch_size() -> None:
    client: StubClient = StubClient()
    _store(client, [], upsert_batch_size=25).upsert(*_chunks(50))

    assert client.attempts == [25, 25]


# --- retry -------------------------------------------------------------------------------

TRANSIENT: list[Callable[[], BaseException]] = [
    _read_timeout,
    lambda: _unexpected(500),
    lambda: _unexpected(503),
    lambda: TimeoutError("slow"),
    lambda: ConnectionError("reset"),
    lambda: ConnectionRefusedError("refused"),
]


@pytest.mark.parametrize("make", TRANSIENT)
def test_transient_error_retried_then_succeeds(make: Callable[[], BaseException]) -> None:
    client, sleeps = StubClient([make(), make()]), []

    _store(client, sleeps).upsert(*_chunks(5))

    assert client.attempts == [5, 5, 5]
    assert client.stored == [5]
    assert sleeps == [1.0, 2.0]


def test_backoff_scales_with_base() -> None:
    client, sleeps = StubClient([TimeoutError(), TimeoutError(), TimeoutError()]), []

    _store(client, sleeps, upsert_attempts=4, retry_backoff_s=0.5).upsert(*_chunks(1))

    assert sleeps == [0.5, 1.0, 2.0]


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (_read_timeout, "ResponseHandlingException(ReadTimeout)"),
        (lambda: _unexpected(502), "UnexpectedResponse(502)"),
        (lambda: TimeoutError(LEAKY_URL), "TimeoutError"),
        (lambda: ConnectionError(LEAKY_URL), "ConnectionError"),
    ],
)
def test_exhaustion_after_three_attempts(make: Callable[[], BaseException], name: str) -> None:
    client, sleeps = StubClient([make(), make(), make(), make()]), []

    with pytest.raises(VectorStoreError) as excinfo:
        _store(client, sleeps).upsert(*_chunks(3))

    assert client.attempts == [3, 3, 3]
    assert sleeps == [1.0, 2.0]
    assert str(excinfo.value) == f"qdrant upsert failed: {name}"
    for fragment in ("qk-SECRET-KEY", "admin", "cluster.internal", "6333", "http"):
        assert fragment not in str(excinfo.value)


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda: _unexpected(400), "UnexpectedResponse(400)"),
        (lambda: _unexpected(404), "UnexpectedResponse(404)"),
        (lambda: _unexpected(429), "UnexpectedResponse(429)"),
        (lambda: ValueError(LEAKY_URL), "ValueError"),
        (lambda: RuntimeError(LEAKY_URL), "RuntimeError"),
    ],
)
def test_non_transient_error_not_retried(make: Callable[[], BaseException], name: str) -> None:
    client, sleeps = StubClient([make()]), []

    with pytest.raises(VectorStoreError) as excinfo:
        _store(client, sleeps).upsert(*_chunks(3))

    assert client.attempts == [3]
    assert sleeps == []
    assert str(excinfo.value) == f"qdrant upsert failed: {name}"
    assert "SECRET" not in str(excinfo.value)


def test_later_batches_run_after_an_earlier_retry_succeeds() -> None:
    client, sleeps = StubClient([TimeoutError()]), []

    _store(client, sleeps).upsert(*_chunks(70))

    assert client.attempts == [32, 32, 32, 6]
    assert client.stored == [32, 32, 6]
    assert sleeps == [1.0]


def test_failure_in_later_batch_keeps_earlier_batches() -> None:
    class FailSecond(StubClient):
        def upsert(self, collection: str, *, points: list[qm.PointStruct], wait: bool) -> None:
            if len(self.attempts) >= 1:
                self.failures = [_unexpected(400)]
            super().upsert(collection, points=points, wait=wait)

    client, sleeps = FailSecond(), []

    with pytest.raises(VectorStoreError):
        _store(client, sleeps).upsert(*_chunks(70))

    assert client.stored == [32]


def test_single_attempt_means_no_retry() -> None:
    client, sleeps = StubClient([TimeoutError()]), []

    with pytest.raises(VectorStoreError, match="TimeoutError"):
        _store(client, sleeps, upsert_attempts=1).upsert(*_chunks(1))

    assert sleeps == []


def test_retry_logged_with_class_names_only(caplog: pytest.LogCaptureFixture) -> None:
    client = StubClient([ConnectionError(LEAKY_URL)])

    with caplog.at_level("WARNING", logger="studiodesk.vectorstore"):
        _store(client, []).upsert(*_chunks(1))

    [record] = [r for r in caplog.records if r.getMessage() == "qdrant upsert failed, retrying"]
    assert record.__dict__["error"] == "ConnectionError"
    assert record.__dict__["attempt"] == 1
    assert record.__dict__["delay_s"] == 1.0
    assert "SECRET" not in caplog.text


def test_exhausted_error_chains_original_cause() -> None:
    cause = ConnectionError("x")
    client = StubClient([cause, cause, cause])

    with pytest.raises(VectorStoreError) as excinfo:
        _store(client, []).upsert(*_chunks(1))

    assert excinfo.value.__cause__ is cause


# --- is_transient ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_read_timeout(), True),
        (ResponseHandlingException(httpx.ConnectError("x")), True),
        (_unexpected(500), True),
        (_unexpected(502), True),
        (_unexpected(599), True),
        (_unexpected(499), False),
        (_unexpected(400), False),
        (_unexpected(404), False),
        (UnexpectedResponse(None, "", b"", httpx.Headers()), False),
        (TimeoutError(), True),
        (ConnectionError(), True),
        (ConnectionResetError(), True),
        (ValueError(), False),
        (RuntimeError(), False),
        (KeyError(), False),
        (VectorStoreError("x"), False),
    ],
)
def test_is_transient_table(exc: BaseException, expected: bool) -> None:
    assert is_transient(exc) is expected
