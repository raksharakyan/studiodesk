"""Free-tier friendly LLM access for evals: a disk cache, a client-side throttle and stats.

`CachedLLM` wraps any `LLMClient`:
- Cache: `evals/.cache/<sha256>.json`, keyed by sha256(model, schema name, system prompt,
  user content). Only the model's parsed output is stored (plus model and schema name for
  debugging), never prompts, keys or headers. A re-run with unchanged prompts costs nothing.
- Throttle: at most `max_rpm` real (non-cached) calls per minute, shared by every wrapper
  that holds the same `Throttle` (answerer and judge use one budget).
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


class Throttle:
    """Spaces calls at least `60 / max_rpm` seconds apart (clock and sleep injectable)."""

    def __init__(
        self,
        max_rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_rpm <= 0:
            raise ValueError("max_rpm must be positive")
        self.interval_s = 60.0 / max_rpm
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self.waited_s = 0.0

    def wait(self) -> None:
        """Block until the next call is allowed, then mark it as started."""
        now = self._clock()
        if self._last is not None:
            remaining = self.interval_s - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                self.waited_s += remaining
        self._last = self._clock()


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
    """`LLMClient` adding the disk cache, the throttle and stats around `inner`."""

    def __init__(
        self,
        inner: LLMClient,
        model: str,
        *,
        cache: ResponseCache | None,
        throttle: Throttle | None,
        stats: CallStats,
        read_cache: bool = True,
    ) -> None:
        """Wrap `inner`; with `read_cache=False` results are refreshed but still stored."""
        self._inner = inner
        self.model = model
        self._cache = cache
        self._throttle = throttle
        self._stats = stats
        self._read_cache = read_cache

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Return the cached output if present, else call the model (throttled) and cache it.

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
        if self._throttle is not None:
            self._throttle.wait()
        self._stats.calls += 1
        try:
            result = self._inner.structured(system, user_content, schema)
        except LLMError:
            self._stats.errors += 1
            raise
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
