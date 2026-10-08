"""Optional Anthropic adapter for `LLMClient` (the default provider is Groq).

`AnthropicLLM` uses the official SDK's
structured outputs: the JSON schema goes in `output_config.format` (built with the SDK's
own `transform_schema`, as `messages.parse` does) together with `output_config.effort`.

Why `beta.messages.create` and not `messages.parse`: `parse` validates the JSON inside the
SDK's response post-parser, so a refused or truncated response would surface as a pydantic
error before `stop_reason` could be inspected. Here `stop_reason` is checked first
(`refusal` -> `LLMRefusal`, `max_tokens` -> `LLMTruncated`) and only then is the text
validated against the schema. The beta path is used because it is the one that accepts
`fallbacks="default"` with the `server-side-fallback-2026-07-01` beta: when the requested
model declines for policy reasons the server retries on its default substitute model.

No `thinking` parameter is sent (omitting it is valid for the default model, while
`{"type": "disabled"}` and `budget_tokens` are rejected), and there is no assistant prefill.
Prompts and model output are never logged; only model, stop reason and token counts are.
"""

import logging
import time
from collections.abc import Callable
from typing import Literal

import anthropic
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaMessage, BetaOutputConfigParam, BetaTextBlock
from pydantic import BaseModel, ValidationError

from studiodesk.config import Settings
from studiodesk.llm.base import (
    LLMInvalidOutput,
    LLMRefusal,
    LLMTruncated,
    LLMUnavailable,
    call_with_retry,
    remaining_s,
    unavailable_from,
)

logger = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
REFUSAL_FALLBACK_BETA: AnthropicBetaParam = "server-side-fallback-2026-07-01"
_OK_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})
_TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AnthropicLLM:
    """`LLMClient` backed by the Anthropic Messages API (structured outputs)."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str,
        max_tokens: int,
        effort: Effort = "medium",
        refusal_fallback: bool = True,
        timeout_s: float = 60.0,
        max_retries: int = 1,
        max_retry_wait_s: float = 10.0,
        request_deadline_s: float = 75.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap an SDK client; the caller owns it (see `close`).

        Args:
            client: Configured SDK client (key, timeout and retries already set).
            model: Model id, e.g. `claude-sonnet-5-5`.
            max_tokens: Upper bound on generated tokens per call.
            effort: `output_config.effort` sent with every call.
            refusal_fallback: Send `fallbacks="default"` with the server-side fallback beta.
            timeout_s: Per-request timeout.
            max_retries: Retries of a transient failure (see `call_with_retry`).
            max_retry_wait_s: Longest wait before a retry; longer waits fail fast.
            request_deadline_s: Budget per `structured` call including all retries.
            sleep: Sleep function (injectable for tests).
            monotonic: Clock for the deadline (injectable for tests).
        """
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._effort: Effort = effort
        self._refusal_fallback = refusal_fallback
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_retry_wait_s = max_retry_wait_s
        self._sleep = sleep
        self._request_deadline_s = request_deadline_s
        self._monotonic = monotonic

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnthropicLLM":
        """Build the SDK client from settings (fixed API host, SDK retries off).

        Raises:
            LLMUnavailable: if no `anthropic_api_key` is configured.
        """
        if settings.anthropic_api_key is None:
            raise LLMUnavailable("anthropic_api_key is not configured")
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            base_url=ANTHROPIC_BASE_URL,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        return cls(
            client,
            model=settings.resolved_llm_model,
            max_tokens=settings.llm_max_tokens,
            effort=settings.llm_effort,
            refusal_fallback=settings.llm_refusal_fallback,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            max_retry_wait_s=settings.llm_max_retry_wait_s,
            request_deadline_s=settings.llm_request_deadline_s,
        )

    def close(self) -> None:
        """Close the underlying SDK client and its connection pool."""
        self._client.close()

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Call the model once and parse its JSON output as `schema`.

        Raises:
            LLMUnavailable: on SDK errors (rate limit, status, connection, timeout).
            LLMRefusal: if the model (and any fallback) refused.
            LLMTruncated: if the output hit the token limit.
            LLMInvalidOutput: if the output is not valid JSON for `schema`.
        """
        deadline = self._monotonic() + self._request_deadline_s
        message = call_with_retry(
            lambda: self._create(system, user_content, schema, deadline),
            max_retries=self._max_retries,
            max_wait_s=self._max_retry_wait_s,
            sleep=self._sleep,
            deadline=deadline,
            monotonic=self._monotonic,
        )
        logger.info(
            "llm call completed",
            extra={
                "model": message.model,
                "stop_reason": message.stop_reason,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )
        _check_stop_reason(message)
        return _parse_output(message, schema)

    def _create(
        self, system: str, user_content: str, schema: type[BaseModel], deadline: float
    ) -> BetaMessage:
        """Send the request (timeout capped by the deadline), SDK errors -> `LLMUnavailable`."""
        timeout = min(self._timeout_s, remaining_s(deadline, self._monotonic))
        output_config: BetaOutputConfigParam = {
            "effort": self._effort,
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)},
        }
        try:
            if self._refusal_fallback:
                return self._client.beta.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_content}],
                    output_config=output_config,
                    fallbacks="default",
                    betas=[REFUSAL_FALLBACK_BETA],
                    timeout=timeout,
                )
            return self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                output_config=output_config,
                timeout=timeout,
            )
        except (
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.APIError,
        ) as exc:
            raise unavailable_from("anthropic", exc, (anthropic.APIConnectionError,)) from exc


def _check_stop_reason(message: BetaMessage) -> None:
    """Raise the matching `LLMError` unless the model finished normally."""
    reason = message.stop_reason
    if reason == "refusal":
        raise LLMRefusal("model refused the request")
    if reason in _TRUNCATED_STOP_REASONS:
        raise LLMTruncated(f"model output truncated: {reason}")
    if reason not in _OK_STOP_REASONS:
        raise LLMInvalidOutput(f"unexpected stop_reason: {reason}")


def _parse_output[SchemaT: BaseModel](message: BetaMessage, schema: type[SchemaT]) -> SchemaT:
    """Validate the concatenated text blocks as JSON for `schema`."""
    text = "".join(block.text for block in message.content if isinstance(block, BetaTextBlock))
    if not text.strip():
        raise LLMInvalidOutput("model returned no text")
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise LLMInvalidOutput(
            f"model output did not match {schema.__name__}: {exc.error_count()} errors"
        ) from None
