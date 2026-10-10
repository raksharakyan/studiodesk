"""Free-tier friendly LLM access for evals: a disk cache, a client-side throttle and stats.

`CachedLLM` wraps any `LLMClient`:
- Cache: `evals/.cache/<sha256>.json`, keyed by sha256(model, schema name, system prompt,
  user content). Only the model's parsed output is stored (plus model and schema name for
  debugging), never prompts, keys or headers. A re-run with unchanged prompts costs nothing.
- Rate limits: one `RateLimiter` per model keeps real (non-cached) calls under both
  `max_rpm` requests and `max_tpm` tokens in any sliding 60 s window. Groq's free tier
  limits tokens per minute per model (8,000 for the models used here), which a pure
  request throttle does not respect. Cache hits cost nothing.
- Stats: real calls, cache hits, errors and token counts. Tokens come from the
  "llm call completed" log records the adapters already emit (no src change needed).
"""

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from studiodesk.llm import LLMClient, LLMError

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
LLM_LOGGER = "studiodesk.llm"
_SEP = "\x00"


def cache_key(model: str, schema_name: str, system: str, user_content: str) -> str:
    """Stable hex sha256 over everything that determines the model's output."""
    payload = _SEP.join((model, schema_name, system, user_content))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """JSON files under `directory`, one per cache key."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> str | None:
        """The cached output JSON for `key`, or None (missing or unreadable entry)."""
        try:
            entry = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        output = entry.get("output") if isinstance(entry, dict) else None
        return json.dumps(output) if output is not None else None

    def put(self, key: str, model: str, schema_name: str, output_json: str) -> None:
        """Store `output_json` atomically (write a temp file, then rename)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"model": model, "schema": schema_name, "output": json.loads(output_json)}
        path = self._path(key)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


WINDOW_S = 60.0
CHARS_PER_TOKEN = 4
_EPSILON_S = 0.05


@dataclass
class _Usage:
    """One call in the window: when it started and the tokens it is charged."""

    started: float
    tokens: int


class RateLimiter:
    """Sliding 60 s window per model that keeps both requests and tokens under limits.

    `acquire(estimate)` waits until one more request fits under `max_rpm` and `estimate`
    more tokens fit under `max_tpm` (a single request larger than `max_tpm` waits for an
    empty window and then goes alone), then charges the estimate. `settle` replaces the
    estimate with the actual usage once the response reports it.

    The output-token estimate starts at the request's `max_output_tokens` and then uses
    the largest output actually seen for this model (never more than the cap), so a few
    long answers keep it conservative without charging 4096 tokens for every call.
    Clock and sleep are injectable; single-threaded use only.
    """

    def __init__(
        self,
        max_rpm: int,
        max_tpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_rpm <= 0 or max_tpm <= 0:
            raise ValueError("max_rpm and max_tpm must be positive")
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._clock = clock
        self._sleep = sleep
        self._window: list[_Usage] = []
        self._max_output_seen: int | None = None
        self.waited_s = 0.0

    def _prune(self, now: float) -> None:
        self._window = [u for u in self._window if u.started > now - WINDOW_S]

    def tokens_in_window(self) -> int:
        """Tokens charged within the last 60 s."""
        self._prune(self._clock())
        return sum(u.tokens for u in self._window)

    def expected_output(self, max_output_tokens: int) -> int:
        """Output tokens to assume for the next call (see class docstring)."""
        if self._max_output_seen is None:
            return max_output_tokens
        return min(self._max_output_seen, max_output_tokens)

    def estimate(self, prompt_chars: int, max_output_tokens: int) -> int:
        """`prompt_chars / 4` plus the expected output tokens."""
        return prompt_chars // CHARS_PER_TOKEN + self.expected_output(max_output_tokens)

    def acquire(self, estimate: int) -> _Usage:
        """Block until a request of `estimate` tokens fits, then charge it."""
        while True:
            now = self._clock()
            self._prune(now)
            used = sum(u.tokens for u in self._window)
            fits_tokens = used + estimate <= self.max_tpm or not self._window
            if len(self._window) < self.max_rpm and fits_tokens:
                usage = _Usage(started=now, tokens=estimate)
                self._window.append(usage)
                return usage
            wait = max(self._window[0].started + WINDOW_S - now, 0.0) + _EPSILON_S
            self._sleep(wait)
            self.waited_s += wait

    def settle(self, usage: _Usage, input_tokens: int, output_tokens: int) -> None:
        """Charge the actual tokens instead of the estimate and learn the output size."""
        usage.tokens = input_tokens + output_tokens
        if output_tokens > 0:
            self._max_output_seen = max(self._max_output_seen or 0, output_tokens)


@dataclass
class CallStats:
    """Cost counters for one run."""

    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    token_reports: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.calls,
            "cache_hits": self.cache_hits,
            "llm_errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls_with_token_counts": self.token_reports,
        }


class CachedLLM:
    """`LLMClient` adding the disk cache, rate limiting and stats around `inner`."""

    def __init__(
        self,
        inner: LLMClient,
        model: str,
        *,
        cache: ResponseCache | None,
        limiter: RateLimiter | None,
        stats: CallStats,
        read_cache: bool = True,
        max_output_tokens: int = 4096,
    ) -> None:
        """Wrap `inner`; with `read_cache=False` results are refreshed but still stored.

        `limiter` should be this model's own `RateLimiter`; `max_output_tokens` is the
        request's output cap (used for the first token estimate).
        """
        self._inner = inner
        self.model = model
        self._cache = cache
        self._limiter = limiter
        self._stats = stats
        self._read_cache = read_cache
        self._max_output_tokens = max_output_tokens

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Return the cached output if present, else call the model (rate limited) and cache it.

        Raises:
            LLMError: if the real call fails (failures are never cached).
        """
        key = cache_key(self.model, schema.__name__, system, user_content)
        if self._cache is not None and self._read_cache:
            cached = self._cache.get(key)
            if cached is not None:
                try:
                    result = schema.model_validate_json(cached)
                except ValidationError:
                    result = None
                if result is not None:
                    self._stats.cache_hits += 1
                    return result
        usage = None
        if self._limiter is not None:
            # The adapters append the JSON schema to the system prompt; count it too.
            chars = len(system) + len(user_content) + len(json.dumps(schema.model_json_schema()))
            usage = self._limiter.acquire(self._limiter.estimate(chars, self._max_output_tokens))
        self._stats.calls += 1
        before = (self._stats.input_tokens, self._stats.output_tokens, self._stats.token_reports)
        try:
            result = self._inner.structured(system, user_content, schema)
        except LLMError:
            self._stats.errors += 1
            raise
        finally:
            # Actual usage comes from the adapter's log records (captured by
            # `capture_tokens`); a schema retry inside the adapter is included. Failed
            # calls that reported usage are corrected too; otherwise the estimate stays.
            reported = self._stats.token_reports > before[2]
            if usage is not None and self._limiter is not None and reported:
                self._limiter.settle(
                    usage,
                    self._stats.input_tokens - before[0],
                    self._stats.output_tokens - before[1],
                )
        if self._cache is not None:
            self._cache.put(key, self.model, schema.__name__, result.model_dump_json())
        return result

    def close(self) -> None:
        """Close the wrapped client."""
        self._inner.close()


class _TokenHandler(logging.Handler):
    """Adds the token counts of "llm call completed" records to `stats`."""

    def __init__(self, stats: CallStats) -> None:
        super().__init__(level=logging.INFO)
        self._stats = stats

    def emit(self, record: logging.LogRecord) -> None:
        tokens_in = getattr(record, "input_tokens", None)
        tokens_out = getattr(record, "output_tokens", None)
        if isinstance(tokens_in, int) or isinstance(tokens_out, int):
            self._stats.token_reports += 1
            self._stats.input_tokens += tokens_in if isinstance(tokens_in, int) else 0
            self._stats.output_tokens += tokens_out if isinstance(tokens_out, int) else 0


@contextmanager
def capture_tokens(stats: CallStats) -> Iterator[None]:
    """Count tokens reported by the LLM adapters' log records while the block runs."""
    logger = logging.getLogger(LLM_LOGGER)
    handler = _TokenHandler(stats)
    previous_level = logger.level
    logger.addHandler(handler)
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
