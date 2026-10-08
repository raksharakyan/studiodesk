"""The provider-neutral `LLMClient` protocol, typed errors and the shared retry policy.

Retry policy (both providers): the SDK clients are built with `max_retries=0` because
neither SDK can cap how long it honours `Retry-After` (Anthropic 1.x waits for any
positive value, Groq for up to 60 s), which would block a request handler. Instead
`call_with_retry` retries a transient failure at most `llm_max_retries` (default 1) times
and only if the wait (`Retry-After`, else a short backoff) is <= `llm_max_retry_wait_s`
(default 10 s). A longer wait fails fast with `LLMUnavailable(retry_after_s=...)`, which
the API returns as 503 with a `Retry-After` header.
"""

import email.utils
import time
from collections.abc import Callable, Mapping
from typing import Protocol

from pydantic import BaseModel

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
INITIAL_BACKOFF_S = 0.5


class LLMError(RuntimeError):
    """Base class for LLM failures. Messages are safe to log, never to return to clients."""


class LLMUnavailable(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The provider could not be reached or rejected the call (rate limit, 5xx, timeout).

    Attributes:
        retryable: True for transient failures (connection, timeout, 408/409/429/5xx).
        retry_after_s: Wait the provider asked for, if it sent one.
    """

    def __init__(
        self, message: str, *, retryable: bool = False, retry_after_s: float | None = None
    ) -> None:
        """Store the safe message and the retry hints."""
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class LLMRefusal(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The model declined to answer."""


class LLMTruncated(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The response hit the token limit before the structured output was complete."""


class LLMInvalidOutput(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The response did not contain JSON matching the requested schema."""


class LLMClient(Protocol):
    """Anything that can turn a system prompt plus user content into a schema instance."""

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Return the model's answer parsed as `schema`.

        Raises:
            LLMError: any failure (a subclass says which kind).
        """
        ...

    def close(self) -> None:
        """Release the underlying client's connections."""
        ...


def error_name(exc: BaseException) -> str:
    """Class name plus HTTP status (if any); never the message, which may echo the request."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}({status})" if isinstance(status, int) else type(exc).__name__


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds to wait from `retry-after-ms` or `retry-after` (seconds or HTTP date)."""
    retry_ms = headers.get("retry-after-ms")
    if retry_ms is not None:
        try:
            return max(float(retry_ms) / 1000, 0.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        parsed = email.utils.parsedate_tz(value)
        if parsed is None:
            return None
        return max(float(email.utils.mktime_tz(parsed) - time.time()), 0.0)


def unavailable_from(
    provider: str, exc: Exception, connection_errors: tuple[type[Exception], ...]
) -> LLMUnavailable:
    """Translate an SDK exception into `LLMUnavailable` with retry hints (no messages)."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    headers: Mapping[str, str] = getattr(response, "headers", None) or {}
    should_retry = headers.get("x-should-retry")
    retryable = isinstance(exc, connection_errors) or (
        isinstance(status, int) and (status in RETRYABLE_STATUS_CODES or status >= 500)
    )
    if should_retry is not None:
        retryable = should_retry.lower() == "true"
    return LLMUnavailable(
        f"{provider} call failed: {error_name(exc)}",
        retryable=retryable,
        retry_after_s=parse_retry_after(headers),
    )


def call_with_retry[T](
    call: Callable[[], T],
    *,
    max_retries: int,
    max_wait_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, retrying transient `LLMUnavailable` errors with a capped wait.

    Raises:
        LLMUnavailable: when not retryable, out of retries, or the wait would exceed
            `max_wait_s` (then `retry_after_s` tells the client how long to wait).
    """
    attempt = 0
    while True:
        try:
            return call()
        except LLMUnavailable as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            wait = (
                exc.retry_after_s
                if exc.retry_after_s is not None
                else INITIAL_BACKOFF_S * 2**attempt
            )
            if wait > max_wait_s:
                raise
            sleep(wait)
            attempt += 1
