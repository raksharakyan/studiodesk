"""Process-wide cap on concurrent LLM calls.

One `LLMConcurrencyGate` (a bounded semaphore) is created per app. `GatedLLM` wraps any
`LLMClient` so each `structured` call holds a slot; if no slot frees up within
`acquire_timeout_s`, the call fails fast with `LLMUnavailable(retry_after_s=...)`, which
the API returns as 503 with a `Retry-After` header. This bounds cost and thread usage
when many requests arrive at once (per process, like the rate limiter).
"""

import threading

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


class GatedLLM:
    """`LLMClient` that runs every call of `inner` inside the shared gate."""

    def __init__(self, inner: LLMClient, gate: LLMConcurrencyGate) -> None:
        """Wrap `inner`; `gate` is shared by all wrappers of the app."""
        self._inner = inner
        self._gate = gate

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Call `inner.structured` while holding a concurrency slot."""
        self._gate.acquire()
        try:
            return self._inner.structured(system, user_content, schema)
        finally:
            self._gate.release()

    def close(self) -> None:
        """The inner client is owned (and closed) by the app lifespan, not the wrapper."""
