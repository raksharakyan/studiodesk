"""Eval LLM wrapper: disk cache, rate limiter, stats and token capture (no network)."""

import json
import logging
from pathlib import Path

import pytest
from evals.llm_cache import (
    CachedLLM,
    CallStats,
    RateLimiter,
    ResponseCache,
    cache_key,
    capture_tokens,
)

from fakes import FakeLLM
from studiodesk.llm import LLMUnavailable
from studiodesk.models.answer import LLMAnswer

ANSWER = LLMAnswer(
    answer="Fixed in 1.0.1 [BUG-0001].", cited_ids=["BUG-0001"], insufficient_context=False
)
FAKE_KEY = "SECRET-MARKER-not-a-real-key-123456"


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _wrapped(
    tmp_path: Path, inner: FakeLLM, *, model: str = "m1", read_cache: bool = True
) -> tuple[CachedLLM, CallStats]:
    stats = CallStats()
    llm = CachedLLM(
        inner,
        model,
        cache=ResponseCache(tmp_path / "cache"),
        limiter=None,
        stats=stats,
        read_cache=read_cache,
    )
    return llm, stats


def test_cache_key_depends_on_every_input() -> None:
    base = cache_key("m", "S", "system", "user")

    assert base == cache_key("m", "S", "system", "user")
    assert len(base) == 64
    assert (
        len(
            {
                base,
                cache_key("m2", "S", "system", "user"),
                cache_key("m", "T", "system", "user"),
                cache_key("m", "S", "system2", "user"),
                cache_key("m", "S", "system", "user2"),
            }
        )
        == 5
    )


def test_cache_key_has_no_field_boundary_ambiguity() -> None:
    assert cache_key("ab", "S", "c", "u") != cache_key("a", "S", "bc", "u")


def test_second_identical_call_is_served_from_cache(tmp_path: Path) -> None:
    inner = FakeLLM({LLMAnswer: ANSWER})
    llm, stats = _wrapped(tmp_path, inner)

    first = llm.structured("sys", "user", LLMAnswer)
    second = llm.structured("sys", "user", LLMAnswer)

    assert first == second == ANSWER
    assert len(inner.calls) == 1
    assert (stats.calls, stats.cache_hits) == (1, 1)


def test_different_model_misses_the_cache(tmp_path: Path) -> None:
    inner = FakeLLM({LLMAnswer: ANSWER})
    a, _ = _wrapped(tmp_path, inner, model="answerer")
    b, _ = _wrapped(tmp_path, inner, model="judge")

    a.structured("sys", "user", LLMAnswer)
    b.structured("sys", "user", LLMAnswer)

    assert len(inner.calls) == 2


def test_no_cache_flag_refreshes_but_still_writes(tmp_path: Path) -> None:
    inner = FakeLLM({LLMAnswer: ANSWER})
    refresh, stats = _wrapped(tmp_path, inner, read_cache=False)
    reader, reader_stats = _wrapped(tmp_path, inner)

    refresh.structured("sys", "user", LLMAnswer)
    refresh.structured("sys", "user", LLMAnswer)
    reader.structured("sys", "user", LLMAnswer)

    assert stats.calls == 2
    assert stats.cache_hits == 0
    assert reader_stats.cache_hits == 1


def test_cache_file_holds_only_the_output(tmp_path: Path) -> None:
    inner = FakeLLM({LLMAnswer: ANSWER})
    llm, _ = _wrapped(tmp_path, inner)

    llm.structured(f"system with {FAKE_KEY}", f"user prompt {FAKE_KEY}", LLMAnswer)

    [entry] = list((tmp_path / "cache").glob("*.json"))
    content = entry.read_text()
    assert FAKE_KEY not in content
    assert "user prompt" not in content
    assert '"schema": "LLMAnswer"' in content
    assert "Fixed in 1.0.1" in content
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_errors_are_counted_and_never_cached(tmp_path: Path) -> None:
    inner = FakeLLM({LLMAnswer: LLMUnavailable("rate limited", retryable=True)})
    llm, stats = _wrapped(tmp_path, inner)

    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            llm.structured("sys", "user", LLMAnswer)

    assert (stats.calls, stats.errors, stats.cache_hits) == (2, 2, 0)
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())


def test_corrupt_cache_entry_falls_back_to_a_real_call(tmp_path: Path) -> None:
    inner = FakeLLM({LLMAnswer: ANSWER})
    llm, stats = _wrapped(tmp_path, inner)
    key = cache_key("m1", "LLMAnswer", "sys", "user")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / f"{key}.json").write_text('{"output": {"answer": 5}}')

    assert llm.structured("sys", "user", LLMAnswer) == ANSWER
    assert (stats.calls, stats.cache_hits) == (1, 0)


def test_unreadable_cache_entry_is_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    (tmp_path / "k.json").write_text("not json")

    assert cache.get("k") is None
    assert cache.get("missing") is None


def test_limiter_caps_requests_per_sliding_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2, 1_000_000, clock=clock, sleep=clock.sleep)

    limiter.acquire(10)
    clock.now += 5
    limiter.acquire(10)
    limiter.acquire(10)  # third request waits until the first leaves the 60 s window

    assert sum(clock.sleeps) == pytest.approx(55.05)


def test_limiter_caps_tokens_per_sliding_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(100, 6_000, clock=clock, sleep=clock.sleep)

    limiter.acquire(4_000)
    clock.now += 10
    limiter.acquire(2_000)  # fits exactly: 6,000
    assert clock.sleeps == []
    limiter.acquire(1)  # waits for the 4,000 entry to expire (t=160)

    assert clock.now == pytest.approx(160.05)
    assert limiter.tokens_in_window() == 2_001


def test_oversized_request_waits_for_an_empty_window_then_goes_alone() -> None:
    clock = FakeClock()
    limiter = RateLimiter(100, 1_000, clock=clock, sleep=clock.sleep)

    limiter.acquire(10)
    limiter.acquire(5_000)

    assert clock.now == pytest.approx(160.05)
    assert limiter.tokens_in_window() == 5_000


def test_settle_replaces_the_estimate_with_actual_usage() -> None:
    clock = FakeClock()
    limiter = RateLimiter(100, 6_000, clock=clock, sleep=clock.sleep)

    usage = limiter.acquire(5_000)
    limiter.settle(usage, 900, 300)
    limiter.acquire(4_000)  # fits once the estimate was corrected to 1,200

    assert clock.sleeps == []
    assert limiter.tokens_in_window() == 5_200


def test_output_estimate_starts_at_cap_then_learns_largest_output() -> None:
    limiter = RateLimiter(10, 10_000)

    assert limiter.estimate(4_000, 4_096) == 1_000 + 4_096
    limiter.settle(limiter.acquire(10), 100, 300)
    limiter.settle(limiter.acquire(10), 100, 700)
    limiter.settle(limiter.acquire(10), 100, 200)

    assert limiter.expected_output(4_096) == 700
    assert limiter.expected_output(500) == 500


@pytest.mark.parametrize(("rpm", "tpm"), [(0, 100), (10, 0), (-1, -1)])
def test_limiter_rejects_non_positive_limits(rpm: int, tpm: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(rpm, tpm)


class ReportingLLM:
    """Inner LLM that logs usage like the real adapters do."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.calls = 0
        self._usage = (input_tokens, output_tokens)

    def structured(self, system: str, user_content: str, schema: type[LLMAnswer]) -> LLMAnswer:
        self.calls += 1
        logging.getLogger("studiodesk.llm.groq_client").info(
            "llm call completed",
            extra={"input_tokens": self._usage[0], "output_tokens": self._usage[1]},
        )
        return ANSWER

    def close(self) -> None:
        """Nothing to release."""


def test_cached_llm_charges_estimate_then_actual_usage(tmp_path: Path) -> None:
    clock = FakeClock()
    limiter = RateLimiter(100, 50_000, clock=clock, sleep=clock.sleep)
    stats = CallStats()
    llm = CachedLLM(
        ReportingLLM(1_000, 250),  # type: ignore[arg-type]
        "m",
        cache=ResponseCache(tmp_path),
        limiter=limiter,
        stats=stats,
        max_output_tokens=4_096,
    )

    with capture_tokens(stats):
        llm.structured("s" * 400, "u" * 400, LLMAnswer)
        llm.structured("s" * 400, "other", LLMAnswer)

    assert limiter.tokens_in_window() == 2 * 1_250
    assert limiter.expected_output(4_096) == 250


def test_estimate_counts_prompt_and_schema(tmp_path: Path) -> None:
    clock = FakeClock()
    limiter = RateLimiter(100, 1_000_000, clock=clock, sleep=clock.sleep)
    llm = CachedLLM(
        FakeLLM({LLMAnswer: ANSWER}),
        "m",
        cache=None,
        limiter=limiter,
        stats=CallStats(),
        max_output_tokens=100,
    )

    llm.structured("s" * 4_000, "u" * 4_000, LLMAnswer)  # no usage reported: estimate stays

    schema_chars = len(json.dumps(LLMAnswer.model_json_schema()))
    assert limiter.tokens_in_window() == (8_000 + schema_chars) // 4 + 100


def test_cached_calls_are_not_rate_limited(tmp_path: Path) -> None:
    clock = FakeClock()
    stats = CallStats()
    limiter = RateLimiter(1, 10, clock=clock, sleep=clock.sleep)
    llm = CachedLLM(
        FakeLLM({LLMAnswer: ANSWER}),
        "m",
        cache=ResponseCache(tmp_path),
        limiter=limiter,
        stats=stats,
    )

    for _ in range(5):
        llm.structured("sys", "user", LLMAnswer)

    assert clock.sleeps == []
    assert stats.cache_hits == 4
    assert len(limiter._window) == 1


def test_capture_tokens_sums_adapter_log_records() -> None:
    stats = CallStats()
    log = logging.getLogger("studiodesk.llm.groq_client")

    with capture_tokens(stats):
        log.info("llm call completed", extra={"input_tokens": 120, "output_tokens": 30})
        log.info("llm call completed", extra={"input_tokens": 80, "output_tokens": None})
        log.info("unrelated record")
    log.info("after the block", extra={"input_tokens": 999, "output_tokens": 999})

    assert (stats.input_tokens, stats.output_tokens, stats.token_reports) == (200, 30, 2)
    assert logging.getLogger("studiodesk.llm").level == logging.NOTSET


def test_stats_as_dict_names() -> None:
    assert set(CallStats().as_dict()) == {
        "llm_calls",
        "cache_hits",
        "llm_errors",
        "input_tokens",
        "output_tokens",
        "calls_with_token_counts",
    }
