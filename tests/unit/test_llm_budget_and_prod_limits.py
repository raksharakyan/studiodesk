"""The process-wide LLM call budget (sliding minute + UTC day), its wiring into GatedLLM,
/ask and /api/demo-info, and the stricter prod rate-limit defaults (multi-limit syntax).
Fixed clocks, fake LLM, in-memory Qdrant; no network."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from fakes import FakeLLM
from studiodesk.config import PROD_RATE_LIMITS, Settings
from studiodesk.embeddings import Embedder
from studiodesk.llm.base import LLMUnavailable
from studiodesk.llm.limits import GatedLLM, LLMCallBudget, LLMConcurrencyGate
from studiodesk.main import create_app
from studiodesk.models.answer import LLMAnswer
from studiodesk.vectorstore import QdrantStore

QDRANT_URL = "https://example.cloud.qdrant.io"
NOON = datetime(2026, 10, 16, 12, 0, 0, tzinfo=UTC)
ANSWER = LLMAnswer(answer="No info.", cited_ids=[], insufficient_context=True)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


# --- LLMCallBudget -------------------------------------------------------------------


def test_per_minute_limit_with_retry_after_until_oldest_expires() -> None:
    clock = Clock(NOON)
    budget = LLMCallBudget(3, 100, clock=clock)
    for seconds in (0, 10, 20):
        clock.now = NOON + timedelta(seconds=seconds)
        budget.take()
    clock.now = NOON + timedelta(seconds=25)
    with pytest.raises(LLMUnavailable) as info:
        budget.take()
    assert info.value.retry_after_s == pytest.approx(35)
    assert budget.remaining_today() == 97  # a refused call takes nothing


def test_sliding_window_frees_slots_exactly_after_60s() -> None:
    clock = Clock(NOON)
    budget = LLMCallBudget(2, 100, clock=clock)
    budget.take()
    clock.now = NOON + timedelta(seconds=30)
    budget.take()
    clock.now = NOON + timedelta(seconds=59, microseconds=999_999)
    with pytest.raises(LLMUnavailable):
        budget.take()
    clock.now = NOON + timedelta(seconds=60)
    budget.take()  # the first call left the window
    with pytest.raises(LLMUnavailable) as info:
        budget.take()
    assert info.value.retry_after_s == pytest.approx(30)


def test_retry_after_is_at_least_one_second() -> None:
    clock = Clock(NOON)
    budget = LLMCallBudget(1, 100, clock=clock)
    budget.take()
    clock.now = NOON + timedelta(seconds=59, microseconds=900_000)
    with pytest.raises(LLMUnavailable) as info:
        budget.take()
    assert info.value.retry_after_s >= 1


def test_daily_limit_until_utc_midnight_then_resets() -> None:
    clock = Clock(datetime(2026, 10, 16, 23, 0, 0, tzinfo=UTC))
    budget = LLMCallBudget(100, 2, clock=clock)
    budget.take()
    clock.now += timedelta(minutes=2)
    budget.take()
    assert budget.remaining_today() == 0
    clock.now += timedelta(minutes=2)
    with pytest.raises(LLMUnavailable) as info:
        budget.take()
    assert info.value.retry_after_s == 56 * 60
    clock.now = datetime(2026, 10, 17, 0, 0, 1, tzinfo=UTC)
    assert budget.remaining_today() == 2
    budget.take()
    assert budget.remaining_today() == 1


def test_daily_limit_checked_before_minute_limit() -> None:
    clock = Clock(datetime(2026, 10, 16, 23, 59, 0, tzinfo=UTC))
    budget = LLMCallBudget(1, 1, clock=clock)
    budget.take()
    with pytest.raises(LLMUnavailable) as info:
        budget.take()
    assert info.value.retry_after_s == 60  # until midnight, not the minute window


@pytest.mark.parametrize(("per_minute", "per_day"), [(0, 1), (1, 0), (-1, 5)])
def test_budget_rejects_non_positive_limits(per_minute: int, per_day: int) -> None:
    with pytest.raises(ValueError):
        LLMCallBudget(per_minute, per_day)


def test_budget_is_thread_safe() -> None:
    budget = LLMCallBudget(1000, 50, clock=Clock(NOON))
    granted: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait()
        for _ in range(20):
            try:
                budget.take()
            except LLMUnavailable:
                continue
            with lock:
                granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(granted) == 50
    assert budget.remaining_today() == 0


def test_from_settings() -> None:
    settings = Settings(_env_file=None, llm_max_calls_per_minute=4, llm_max_calls_per_day=9)
    assert LLMCallBudget.from_settings(settings).remaining_today() == 9


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_max_calls_per_minute", 0),
        ("llm_max_calls_per_minute", 10_001),
        ("llm_max_calls_per_day", 0),
        ("llm_max_calls_per_day", 1_000_001),
    ],
)
def test_budget_settings_bounds(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})  # type: ignore[arg-type]


def test_budget_defaults() -> None:
    settings = Settings(_env_file=None)
    assert (settings.llm_max_calls_per_minute, settings.llm_max_calls_per_day) == (6, 300)


# --- GatedLLM ------------------------------------------------------------------------


class Out(BaseModel):
    value: int


def test_gated_llm_takes_one_unit_per_call_and_releases_the_gate() -> None:
    inner = FakeLLM({Out: Out(value=1)})
    gate = LLMConcurrencyGate(1, acquire_timeout_s=0.01, busy_retry_after_s=5)
    budget = LLMCallBudget(2, 100, clock=Clock(NOON))
    llm = GatedLLM(inner, gate, budget)
    llm.structured("s", "u", Out)
    llm.structured("s", "u", Out)
    with pytest.raises(LLMUnavailable):
        llm.structured("s", "u", Out)
    assert len(inner.calls) == 2  # the provider is not called over budget
    # The gate slot was released after the budget refusal: another wrapper can still acquire.
    gate.acquire()
    gate.release()


def test_gated_llm_without_budget_is_unlimited() -> None:
    inner = FakeLLM({Out: Out(value=1)})
    llm = GatedLLM(inner, LLMConcurrencyGate(1, acquire_timeout_s=0.01, busy_retry_after_s=5))
    for _ in range(20):
        llm.structured("s", "u", Out)
    assert len(inner.calls) == 20


# --- through the app -----------------------------------------------------------------


@contextmanager
def app_client(
    store: QdrantStore, embedder: Embedder, llm: FakeLLM, **fields: Any
) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, **{"app_env": "test", **fields})
    with TestClient(create_app(settings, embedder=embedder, store=store, llm=llm)) as client:
        yield client


def test_ask_returns_503_with_retry_after_over_budget(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    llm = FakeLLM({LLMAnswer: ANSWER})
    with app_client(
        ingested_store, fake_embedder, llm, llm_max_calls_per_minute=2, ask_rate_limit="100/minute"
    ) as client:
        statuses = [
            client.post("/ask", json={"question": "Why does my save break?"}).status_code
            for _ in range(2)
        ]
        refused = client.post("/ask", json={"question": "Why does my save break?"})
        remaining = client.get("/api/demo-info").json()["llm"]["remaining_today"]
    assert statuses == [200, 200]
    assert refused.status_code == 503
    assert int(refused.headers["retry-after"]) >= 1
    assert "budget" not in refused.text.lower()  # generic detail only
    assert len(llm.calls) == 2
    assert remaining == 300 - 2


def test_demo_info_reports_llm_remaining(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder, FakeLLM({}), llm_max_calls_per_day=7) as client:
        body = client.get("/api/demo-info").json()
    assert body["llm"] == {"remaining_today": 7}


# --- rate limit syntax and prod defaults ---------------------------------------------


@pytest.mark.parametrize(
    "value", ["10/minute", "3/minute;30/day", "1/second;2/minute;3/hour;4/day"]
)
def test_multi_limit_syntax_accepted(value: str) -> None:
    assert Settings(_env_file=None, ask_rate_limit=value).ask_rate_limit == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        ";",
        "3/minute;",
        ";3/minute",
        "3/minute;;30/day",
        "3/minute, 30/day",
        "3/minute;30/week",
        "0/minute",
        "1/second;2/minute;3/hour;4/day;5/day",
        "3 / minute",
    ],
)
def test_bad_limit_syntax_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ask_rate_limit=value)


def test_prod_uses_strict_rate_limits_by_default() -> None:
    settings = Settings(_env_file=None, app_env="prod", qdrant_url=QDRANT_URL)
    for name, value in PROD_RATE_LIMITS.items():
        assert getattr(settings, name) == value
    assert PROD_RATE_LIMITS == {
        "ask_rate_limit": "3/minute;30/day",
        "bugs_rate_limit": "3/minute;20/day",
        "actions_rate_limit": "2/minute;5/day",
        "voice_rate_limit": "2/minute;5/day",
    }


def test_prod_keeps_explicit_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = Settings(
        _env_file=None, app_env="prod", qdrant_url=QDRANT_URL, ask_rate_limit="50/minute"
    )
    assert explicit.ask_rate_limit == "50/minute"
    assert explicit.bugs_rate_limit == PROD_RATE_LIMITS["bugs_rate_limit"]
    monkeypatch.setenv("VOICE_RATE_LIMIT", "9/minute")
    from_env = Settings(_env_file=None, app_env="prod", qdrant_url=QDRANT_URL)
    assert from_env.voice_rate_limit == "9/minute"


@pytest.mark.parametrize("env", ["dev", "test"])
def test_non_prod_keeps_relaxed_defaults(env: str) -> None:
    settings = Settings(_env_file=None, app_env=env)
    assert settings.ask_rate_limit == "10/minute"
    assert settings.voice_rate_limit == "10/minute"


def test_prod_multi_limit_enforced_per_minute(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    llm = FakeLLM({LLMAnswer: ANSWER})
    with app_client(
        ingested_store,
        fake_embedder,
        llm,
        app_env="prod",
        qdrant_url=QDRANT_URL,
        llm_max_calls_per_minute=100,
    ) as client:
        statuses = [
            client.post("/ask", json={"question": "Why does my save break?"}).status_code
            for _ in range(4)
        ]
    assert statuses == [200, 200, 200, 429]
