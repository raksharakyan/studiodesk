"""Groq adapter for `LLMClient` (the default, free-tier provider).

Groq's JSON mode (`response_format={"type": "json_object"}`) guarantees syntactically
valid JSON but not a schema, so the target JSON schema (`model_json_schema()`) is appended
to the system prompt and the reply is validated with Pydantic. On a validation failure
the call is retried exactly once with the validation errors appended to the user message
(error locations and messages only, never the offending input); a second failure raises
`LLMInvalidOutput`, which the API returns as a generic 502.

The default model `openai/gpt-oss-120b` is a reasoning model: `reasoning_effort` comes from
settings and `include_reasoning=False` keeps reasoning out of the response; only
`message.content` is ever parsed.

Sampling uses a low temperature from settings, `max_tokens` is capped from settings and
each request carries the configured timeout. Transient failures follow the capped retry
policy in `studiodesk.llm.base` (SDK retries are off). The API host is pinned so a
`GROQ_BASE_URL` in the environment cannot redirect traffic. Prompts and model output are
never logged; only model, finish reason and token counts are.
"""

import json
import logging
import time
from collections.abc import Callable
from typing import Literal

import groq
from groq.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from studiodesk.config import Settings
from studiodesk.llm.base import (
    LLMInvalidOutput,
    LLMTruncated,
    LLMUnavailable,
    call_with_retry,
    remaining_s,
    unavailable_from,
)

logger = logging.getLogger(__name__)

ReasoningEffort = Literal["low", "medium", "high"]

GROQ_BASE_URL = "https://api.groq.com"
MAX_ERRORS_IN_RETRY = 10

SCHEMA_INSTRUCTIONS = (
    "\n\nOutput format: reply with a single JSON object, and nothing else, that conforms to "
    "this JSON schema:\n"
)
RETRY_INSTRUCTIONS = (
    "\n\nYour previous reply did not match the required JSON schema. Validation errors:\n"
)


def schema_system_prompt(system: str, schema: type[BaseModel]) -> str:
    """Append the schema to the system prompt (deterministic: sorted keys, no timestamps)."""
    return system + SCHEMA_INSTRUCTIONS + json.dumps(schema.model_json_schema(), sort_keys=True)


def validation_summary(exc: ValidationError) -> str:
    """One line per error (location and message only; input values are never included)."""
    lines = [
        f"- {'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)[:MAX_ERRORS_IN_RETRY]
    ]
    return "\n".join(lines)


class GroqLLM:
    """`LLMClient` backed by Groq chat completions in JSON mode."""

    def __init__(
        self,
        client: groq.Groq,
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.1,
        reasoning_effort: ReasoningEffort = "medium",
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
            model: Model id, e.g. `openai/gpt-oss-120b`.
            max_tokens: Upper bound on generated tokens per call.
            temperature: Sampling temperature (keep it low for consistent judgements).
            reasoning_effort: Reasoning effort, sent with `include_reasoning=False`.
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
        self._temperature = temperature
        self._reasoning_effort: ReasoningEffort = reasoning_effort
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._max_retry_wait_s = max_retry_wait_s
        self._sleep = sleep
        self._request_deadline_s = request_deadline_s
        self._monotonic = monotonic

    @classmethod
    def from_settings(cls, settings: Settings) -> "GroqLLM":
        """Build the SDK client from settings (fixed API host, SDK retries off).

        Raises:
            LLMUnavailable: if no `groq_api_key` is configured.
        """
        if settings.groq_api_key is None:
            raise LLMUnavailable("groq_api_key is not configured")
        client = groq.Groq(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=GROQ_BASE_URL,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        return cls(
            client,
            model=settings.resolved_llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            reasoning_effort=settings.llm_reasoning_effort,
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
        """Call the model and validate its JSON as `schema`, retrying once on mismatch.

        Raises:
            LLMUnavailable: on SDK errors (rate limit, status, connection, timeout).
            LLMTruncated: if the output hit the token limit.
            LLMInvalidOutput: if the output fails validation twice (or is empty).
        """
        system_prompt = schema_system_prompt(system, schema)
        deadline = self._monotonic() + self._request_deadline_s
        try:
            return schema.model_validate_json(self._complete(system_prompt, user_content, deadline))
        except ValidationError as exc:
            logger.warning(
                "llm output failed validation, retrying once",
                extra={"schema": schema.__name__, "errors": exc.error_count()},
            )
            retry_content = user_content + RETRY_INSTRUCTIONS + validation_summary(exc)
        if self._monotonic() >= deadline:
            raise LLMInvalidOutput(
                f"model output did not match {schema.__name__}; deadline left no retry"
            )
        try:
            return schema.model_validate_json(
                self._complete(system_prompt, retry_content, deadline)
            )
        except ValidationError as exc:
            raise LLMInvalidOutput(
                f"model output did not match {schema.__name__} after retry: "
                f"{exc.error_count()} errors"
            ) from None

    def _complete(self, system: str, user_content: str, deadline: float) -> str:
        """Send one JSON-mode request (with the capped retry policy); return the text."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        completion = call_with_retry(
            lambda: self._request(messages, deadline),
            max_retries=self._max_retries,
            max_wait_s=self._max_retry_wait_s,
            sleep=self._sleep,
            deadline=deadline,
            monotonic=self._monotonic,
        )
        return _message_text(completion)

    def _request(
        self, messages: list[ChatCompletionMessageParam], deadline: float
    ) -> ChatCompletion:
        """One HTTP request (timeout capped by the deadline), SDK errors -> `LLMUnavailable`."""
        timeout = min(self._timeout_s, remaining_s(deadline, self._monotonic))
        try:
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self._temperature,
                max_completion_tokens=self._max_tokens,
                reasoning_effort=self._reasoning_effort,
                include_reasoning=False,
                timeout=timeout,
            )
        except (
            groq.RateLimitError,
            groq.APIStatusError,
            groq.APITimeoutError,
            groq.APIConnectionError,
            groq.APIError,
        ) as exc:
            raise unavailable_from("groq", exc, (groq.APIConnectionError,)) from exc


def _message_text(completion: ChatCompletion) -> str:
    """Check the finish reason and return the first choice's `message.content` only.

    Any `message.reasoning` (reasoning models) is ignored, never parsed as the answer.
    """
    usage = completion.usage
    choice = completion.choices[0] if completion.choices else None
    logger.info(
        "llm call completed",
        extra={
            "model": completion.model,
            "finish_reason": choice.finish_reason if choice else None,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
        },
    )
    if choice is None:
        raise LLMInvalidOutput("model returned no choices")
    if choice.finish_reason == "length":
        raise LLMTruncated("model output truncated: length")
    if choice.finish_reason != "stop":
        raise LLMInvalidOutput(f"unexpected finish_reason: {choice.finish_reason}")
    content = choice.message.content
    if not content or not content.strip():
        raise LLMInvalidOutput("model returned no text")
    return content
