"""Process-wide caps on concurrent LLM calls and on the LLM call budget.

One `LLMConcurrencyGate` (a bounded semaphore) is created per app. `GatedLLM` wraps any
`LLMClient` so each `structured` call holds a slot; if no slot frees up within
`acquire_timeout_s`, the call fails fast with `LLMUnavailable(retry_after_s=...)`, which
the API returns as 503 with a `Retry-After` header. This bounds cost and thread usage
when many requests arrive at once (per process, like the rate limiter).

`LLMCallBudget` caps how many `structured` calls the whole app makes per sliding minute
and per UTC day, across all clients (a global spend ceiling on top of the per-IP rate
limits). It counts one unit per `structured` call, taken after the concurrency slot and
right before the provider is called. A call can reach the provider more than once inside
the adapter (transient retries, `llm_max_retries`, and Groq's one schema retry), so the
worst case is `(1 + llm_max_retries) * 2` provider requests per unit. Over budget, the
call fails with `LLMUnavailable(retry_after_s=...)` -> 503 + Retry-After.
"""

import math
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta

from pydantic import BaseModel

from studiodesk.config import Settings
from studiodesk.llm.base import LLMClient, LLMUnavailable


class LLMConcurrencyGate:
    """A bounded semaphore with an acquire timeout and a client back-off hint."""

    def __init__(
        self, max_concurrency: int, *, acquire_timeout_s: float, busy_retry_after_s: float
    ) -> None:
        """Allow `max_concurrency` concurrent calls; wait at most `acquire_timeout_s`."""
        if max_concurrency <= 0 or acquire_timeout_s < 0 or busy_retry_after_s <= 0:
            raise ValueError("invalid concurrency gate settings")
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._acquire_timeout_s = acquire_timeout_s
        self._busy_retry_after_s = busy_retry_after_s

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMConcurrencyGate":
        """Build the gate from `llm_max_concurrency` and the busy settings."""
        return cls(
            settings.llm_max_concurrency,
            acquire_timeout_s=settings.llm_concurrency_wait_s,
            busy_retry_after_s=settings.llm_busy_retry_after_s,
        )

    def acquire(self) -> None:
        """Take a slot or raise `LLMUnavailable` (with `retry_after_s`) after the timeout."""
        if not self._semaphore.acquire(timeout=self._acquire_timeout_s):
            raise LLMUnavailable(
                "llm concurrency limit reached", retry_after_s=self._busy_retry_after_s
            )

    def release(self) -> None:
        """Return a slot taken by `acquire`."""
        self._semaphore.release()


WINDOW = timedelta(minutes=1)


def _utc_now() -> datetime:
    """Current aware UTC time (the default budget clock)."""
    return datetime.now(UTC)


def _seconds_until_utc_midnight(now: datetime) -> int:
    """Whole seconds from aware UTC `now` to the next UTC midnight (at least 1)."""
    midnight = datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=now.tzinfo)
    return max(math.ceil((midnight - now).total_seconds()), 1)


class LLMCallBudget:
    """Thread-safe process-wide LLM call budget: a sliding minute plus a UTC-day count."""

    def __init__(
        self,
        per_minute: int,
        per_day: int,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Allow `per_minute` calls in any 60 s window and `per_day` per UTC day.

        Args:
            per_minute: Maximum calls in any sliding 60-second window; must be positive.
            per_day: Maximum calls per UTC day; must be positive.
            clock: Returns the current aware UTC datetime (injectable for tests).
        """
        if per_minute <= 0 or per_day <= 0:
            raise ValueError("per_minute and per_day must be positive")
        self._per_minute = per_minute
        self._per_day = per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: deque[datetime] = deque()
        self._day: date = clock().date()
        self._used_today = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMCallBudget":
        """Build the budget from `llm_max_calls_per_minute` and `llm_max_calls_per_day`."""
        return cls(settings.llm_max_calls_per_minute, settings.llm_max_calls_per_day)

    def take(self) -> None:
        """Count one call, or raise `LLMUnavailable` with the wait until one is allowed."""
        with self._lock:
            now = self._clock()
            self._roll(now)
            if self._used_today >= self._per_day:
                raise LLMUnavailable(
                    "llm daily call budget reached",
                    retry_after_s=_seconds_until_utc_midnight(now),
                )
            if len(self._recent) >= self._per_minute:
                wait = (self._recent[0] + WINDOW - now).total_seconds()
                raise LLMUnavailable(
                    "llm per-minute call budget reached", retry_after_s=max(wait, 1.0)
                )
            self._recent.append(now)
            self._used_today += 1

    def remaining_today(self) -> int:
        """Calls still allowed today (read-only; never negative)."""
        with self._lock:
            self._roll(self._clock())
            return max(self._per_day - self._used_today, 0)

    def _roll(self, now: datetime) -> None:
        """Drop calls older than the window; reset the day count on a new UTC day."""
        while self._recent and self._recent[0] <= now - WINDOW:
            self._recent.popleft()
        if now.date() != self._day:
            self._day = now.date()
            self._used_today = 0


class GatedLLM:
    """`LLMClient` that runs every call of `inner` inside the shared gate and budget."""

    def __init__(
        self, inner: LLMClient, gate: LLMConcurrencyGate, budget: LLMCallBudget | None = None
    ) -> None:
        """Wrap `inner`; `gate` and `budget` are shared by all wrappers of the app."""
        self._inner = inner
        self._gate = gate
        self._budget = budget

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Call `inner.structured` while holding a concurrency slot, within the budget."""
        self._gate.acquire()
        try:
            if self._budget is not None:
                self._budget.take()
            return self._inner.structured(system, user_content, schema)
        finally:
            self._gate.release()

    def close(self) -> None:
        """The inner client is owned (and closed) by the app lifespan, not the wrapper."""
