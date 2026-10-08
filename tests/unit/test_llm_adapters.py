"""Groq and Anthropic adapters against MockTransport SDK clients, plus the shared retry policy.

No network: Groq's SDK gets an `httpx` MockTransport, Anthropic's an `httpx2` one. The
`from_settings` paths are exercised by swapping the SDK constructor for one that injects the
mock transport, so the real base URL the adapter chose is what the handler sees.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

import anthropic
import groq
import httpx
import httpx2
import pytest

from studiodesk.config import Settings
from studiodesk.llm import (
    AnthropicLLM,
    GroqLLM,
    LLMInvalidOutput,
    LLMRefusal,
    LLMTruncated,
    LLMUnavailable,
    build_llm,
)
from studiodesk.llm import anthropic_client as anthropic_module
from studiodesk.llm import groq_client as groq_module
from studiodesk.llm.base import call_with_retry, parse_retry_after
from studiodesk.llm.groq_client import RETRY_INSTRUCTIONS, SCHEMA_INSTRUCTIONS
from studiodesk.models.answer import LLMAnswer

VALID = '{"answer": "ok", "cited_ids": ["BUG-0001"], "insufficient_context": false}'
FAKE_KEY = "sk-test-key-not-real-123456"
LLM_ENV_VARS = (
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "GROQ_BASE_URL",
    "ANTHROPIC_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- Groq helpers


def groq_completion(
    content: str | None = VALID, finish_reason: str = "stop", **message: Any
) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, **message},
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class Recorder:
    """MockTransport handler replaying a script of responses (or exceptions)."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        item = self.script[min(len(self.requests), len(self.script)) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def groq_llm(script: list[Any], sleeps: list[float] | None = None, **kwargs: Any) -> tuple:
    rec = Recorder(script)
    client = groq.Groq(
        api_key=FAKE_KEY,
        base_url="https://api.groq.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(rec)),
    )
    params: dict[str, Any] = {"model": "openai/gpt-oss-120b", "max_tokens": 321}
    params.update(kwargs)
    sleep_log = sleeps if sleeps is not None else []
    return GroqLLM(client, sleep=sleep_log.append, **params), rec


def ok(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=payload)


# --------------------------------------------------------------------------- Groq tests


def test_groq_request_body_shape() -> None:
    llm, rec = groq_llm(
        [ok(groq_completion())], temperature=0.15, reasoning_effort="high", timeout_s=7
    )

    llm.structured("SYSTEM PROMPT", "USER CONTENT", LLMAnswer)

    [request] = rec.requests
    assert request.method == "POST"
    assert request.url.host == "api.groq.com"
    assert request.url.path == "/openai/v1/chat/completions"
    body = rec.bodies()[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.15
    assert body["max_completion_tokens"] == 321
    assert body["reasoning_effort"] == "high"
    assert body["include_reasoning"] is False
    assert body["model"] == "openai/gpt-oss-120b"
    system, user = body["messages"]
    assert system["role"] == "system"
    assert system["content"].startswith("SYSTEM PROMPT" + SCHEMA_INSTRUCTIONS)
    assert '"cited_ids"' in system["content"]  # schema is appended
    assert user == {"role": "user", "content": "USER CONTENT"}


def test_groq_ignores_reasoning_field() -> None:
    leaked = '{"answer": "LEAKED", "cited_ids": [], "insufficient_context": true}'
    llm, _ = groq_llm([ok(groq_completion(VALID, reasoning=leaked))])

    assert llm.structured("S", "U", LLMAnswer).answer == "ok"


def test_groq_validation_retry_once_then_succeeds() -> None:
    bad = '{"answer": "SECRET-BAD-INPUT", "cited_ids": "nope"}'
    llm, rec = groq_llm([ok(groq_completion(bad)), ok(groq_completion(VALID))])

    assert llm.structured("S", "U", LLMAnswer).answer == "ok"

    assert len(rec.requests) == 2
    retry_user = rec.bodies()[1]["messages"][1]["content"]
    assert retry_user.startswith("U" + RETRY_INSTRUCTIONS)
    assert "cited_ids" in retry_user
    assert "SECRET-BAD-INPUT" not in retry_user  # input values never echoed back


def test_groq_validation_fails_twice_raises_invalid_output() -> None:
    llm, rec = groq_llm([ok(groq_completion('{"x": 1}')), ok(groq_completion("not json"))])

    with pytest.raises(LLMInvalidOutput):
        llm.structured("S", "U", LLMAnswer)

    assert len(rec.requests) == 2


def test_groq_finish_reason_length_is_truncated_without_retry() -> None:
    llm, rec = groq_llm([ok(groq_completion('{"answer": "par', finish_reason="length"))])

    with pytest.raises(LLMTruncated):
        llm.structured("S", "U", LLMAnswer)

    assert len(rec.requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        groq_completion(""),
        groq_completion("   "),
        groq_completion(None),
        groq_completion(VALID, finish_reason="content_filter"),
        {**groq_completion(), "choices": []},
    ],
)
def test_groq_empty_or_odd_responses_are_invalid_output(payload: dict[str, Any]) -> None:
    llm, _ = groq_llm([ok(payload)])
    with pytest.raises(LLMInvalidOutput):
        llm.structured("S", "U", LLMAnswer)


def test_groq_429_short_retry_after_retries_once() -> None:
    sleeps: list[float] = []
    llm, rec = groq_llm(
        [
            httpx.Response(429, headers={"retry-after": "10"}, json={"error": {}}),
            ok(groq_completion()),
        ],
        sleeps,
    )

    assert llm.structured("S", "U", LLMAnswer).answer == "ok"
    assert sleeps == [10.0]
    assert len(rec.requests) == 2


def test_groq_429_long_retry_after_fails_fast() -> None:
    sleeps: list[float] = []
    llm, rec = groq_llm(
        [httpx.Response(429, headers={"retry-after": "10.5"}, json={"error": {}})], sleeps
    )

    with pytest.raises(LLMUnavailable) as info:
        llm.structured("S", "U", LLMAnswer)

    assert info.value.retry_after_s == 10.5
    assert info.value.retryable is True
    assert sleeps == []
    assert len(rec.requests) == 1


def test_groq_429_twice_gives_up_after_one_retry() -> None:
    sleeps: list[float] = []
    resp = httpx.Response(429, headers={"retry-after": "1"}, json={"error": {}})
    llm, rec = groq_llm([resp, resp, resp], sleeps)

    with pytest.raises(LLMUnavailable):
        llm.structured("S", "U", LLMAnswer)

    assert len(rec.requests) == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    ("response", "retryable", "calls"),
    [
        (httpx.Response(500, json={"error": {"message": "boom " + FAKE_KEY}}), True, 2),
        (httpx.Response(503, json={"error": {}}), True, 2),
        (httpx.Response(401, json={"error": {"message": "bad key " + FAKE_KEY}}), False, 1),
        (httpx.Response(400, json={"error": {}}), False, 1),
        (httpx.Response(404, json={"error": {}}), False, 1),
        (httpx.ConnectError("refused"), True, 2),
        (httpx.ReadTimeout("slow"), True, 2),
    ],
)
def test_groq_sdk_errors_are_mapped(response: Any, retryable: bool, calls: int) -> None:
    sleeps: list[float] = []
    llm, rec = groq_llm([response, response], sleeps)

    with pytest.raises(LLMUnavailable) as info:
        llm.structured("S", "U", LLMAnswer)

    assert info.value.retryable is retryable
    assert len(rec.requests) == calls
    assert FAKE_KEY not in str(info.value)
    assert "boom" not in str(info.value) and "bad key" not in str(info.value)
    assert str(info.value).startswith("groq call failed: ")


def test_groq_x_should_retry_header_overrides() -> None:
    llm, rec = groq_llm([httpx.Response(500, headers={"x-should-retry": "false"}, json={})])
    with pytest.raises(LLMUnavailable) as info:
        llm.structured("S", "U", LLMAnswer)
    assert info.value.retryable is False
    assert len(rec.requests) == 1


def test_groq_logs_never_contain_prompt_or_output(caplog: pytest.LogCaptureFixture) -> None:
    llm, _ = groq_llm([ok(groq_completion(VALID))])
    with caplog.at_level(logging.DEBUG, logger="studiodesk"):
        llm.structured("SYSTEM-MARKER", "USER-MARKER", LLMAnswer)

    for record in caplog.records:
        flat = json.dumps(record.__dict__, default=str)
        assert "SYSTEM-MARKER" not in flat and "USER-MARKER" not in flat
        assert '"answer": "ok"' not in flat and FAKE_KEY not in flat


def _patch_sdk(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    attr: str,
    transport: Callable[[], Any],
) -> list[dict[str, Any]]:
    """Make `module.<sdk>.<attr>(...)` inject our mock transport; record constructor kwargs."""
    original = getattr(module, attr)
    seen: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return original(**kwargs, http_client=transport())

    monkeypatch.setattr(module, attr, factory)
    return seen


def test_groq_from_settings_pins_host_despite_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_BASE_URL", "https://evil.example.com")
    rec = Recorder([ok(groq_completion())])
    seen = _patch_sdk(
        monkeypatch,
        groq_module.groq,
        "Groq",
        lambda: httpx.Client(transport=httpx.MockTransport(rec)),
    )
    settings = Settings(_env_file=None, groq_api_key=FAKE_KEY, llm_max_tokens=77)

    llm = build_llm(settings)
    assert isinstance(llm, GroqLLM)
    llm.structured("S", "U", LLMAnswer)
    llm.close()

    assert seen[0]["max_retries"] == 0
    assert rec.requests[0].url.host == "api.groq.com"
    assert rec.requests[0].url.scheme == "https"
    assert rec.bodies()[0]["max_completion_tokens"] == 77
    assert rec.bodies()[0]["model"] == "openai/gpt-oss-120b"


def test_groq_from_settings_without_key_raises() -> None:
    with pytest.raises(LLMUnavailable):
        GroqLLM.from_settings(Settings(_env_file=None))


# --------------------------------------------------------------------------- Anthropic


def anthropic_message(
    text: str = VALID, stop_reason: str = "end_turn", content: list[Any] | None = None
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }


def anthropic_llm(script: list[Any], sleeps: list[float] | None = None, **kwargs: Any) -> tuple:
    rec = Recorder(script)
    client = anthropic.Anthropic(
        api_key=FAKE_KEY,
        base_url="https://api.anthropic.com",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(rec)),
    )
    params: dict[str, Any] = {"model": "claude-sonnet-5-5", "max_tokens": 222}
    params.update(kwargs)
    sleep_log = sleeps if sleeps is not None else []
    return AnthropicLLM(client, sleep=sleep_log.append, **params), rec


def a_ok(payload: dict[str, Any]) -> httpx2.Response:
    return httpx2.Response(200, json=payload)


def test_anthropic_request_with_fallback_enabled() -> None:
    llm, rec = anthropic_llm([a_ok(anthropic_message())], effort="high")

    assert llm.structured("SYS", "USR", LLMAnswer).answer == "ok"

    [request] = rec.requests
    assert request.url.host == "api.anthropic.com"
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    body = rec.bodies()[0]
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "high"
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["properties"]) == {"answer", "cited_ids", "insufficient_context"}
    assert "thinking" not in body
    assert body["max_tokens"] == 222
    assert body["system"] == "SYS"
    assert body["messages"] == [{"role": "user", "content": "USR"}]
    assert len(body["messages"]) == 1  # no assistant prefill


def test_anthropic_request_with_fallback_disabled() -> None:
    llm, rec = anthropic_llm([a_ok(anthropic_message())], refusal_fallback=False)

    llm.structured("SYS", "USR", LLMAnswer)

    request = rec.requests[0]
    assert "fallback" not in request.headers.get("anthropic-beta", "")
    body = rec.bodies()[0]
    assert "fallbacks" not in body
    assert "thinking" not in body
    assert body["output_config"]["effort"] == "medium"


@pytest.mark.parametrize(
    ("stop_reason", "text", "error"),
    [
        ("refusal", VALID, LLMRefusal),
        ("refusal", "I can't help with that", LLMRefusal),
        ("max_tokens", '{"answer": "trunc', LLMTruncated),
        ("max_tokens", VALID, LLMTruncated),
        ("model_context_window_exceeded", VALID, LLMTruncated),
        ("tool_use", VALID, LLMInvalidOutput),
        ("pause_turn", VALID, LLMInvalidOutput),
    ],
)
def test_anthropic_stop_reason_checked_before_parsing(
    stop_reason: str, text: str, error: type[Exception]
) -> None:
    llm, rec = anthropic_llm([a_ok(anthropic_message(text, stop_reason))])

    with pytest.raises(error):
        llm.structured("S", "U", LLMAnswer)

    assert len(rec.requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        anthropic_message("not json"),
        anthropic_message('{"answer": 1}'),
        anthropic_message("   "),
        anthropic_message(content=[]),
    ],
)
def test_anthropic_bad_output_is_invalid_without_retry(payload: dict[str, Any]) -> None:
    llm, rec = anthropic_llm([a_ok(payload)])
    with pytest.raises(LLMInvalidOutput) as info:
        llm.structured("S", "U", LLMAnswer)
    assert len(rec.requests) == 1
    assert "not json" not in str(info.value)


def test_anthropic_concatenates_text_blocks() -> None:
    blocks = [{"type": "text", "text": VALID[:10]}, {"type": "text", "text": VALID[10:]}]
    llm, _ = anthropic_llm([a_ok(anthropic_message(content=blocks))])
    assert llm.structured("S", "U", LLMAnswer).cited_ids == ["BUG-0001"]


def test_anthropic_retry_policy() -> None:
    sleeps: list[float] = []
    llm, rec = anthropic_llm(
        [
            httpx2.Response(429, headers={"retry-after": "3"}, json={"type": "error"}),
            a_ok(anthropic_message()),
        ],
        sleeps,
    )
    assert llm.structured("S", "U", LLMAnswer).answer == "ok"
    assert sleeps == [3.0]
    assert len(rec.requests) == 2


def test_anthropic_long_retry_after_fails_fast() -> None:
    sleeps: list[float] = []
    llm, rec = anthropic_llm(
        [httpx2.Response(429, headers={"retry-after": "60"}, json={"type": "error"})], sleeps
    )
    with pytest.raises(LLMUnavailable) as info:
        llm.structured("S", "U", LLMAnswer)
    assert info.value.retry_after_s == 60.0
    assert sleeps == []
    assert len(rec.requests) == 1


@pytest.mark.parametrize(
    ("response", "retryable"),
    [
        (httpx2.Response(529, json={"type": "error", "error": {"message": FAKE_KEY}}), True),
        (httpx2.Response(500, json={"type": "error"}), True),
        (httpx2.Response(401, json={"type": "error", "error": {"message": FAKE_KEY}}), False),
        (httpx2.Response(400, json={"type": "error"}), False),
        (httpx2.ConnectError("refused"), True),
        (httpx2.ReadTimeout("slow"), True),
    ],
)
def test_anthropic_sdk_errors_are_mapped(response: Any, retryable: bool) -> None:
    llm, _ = anthropic_llm([response, response])
    with pytest.raises(LLMUnavailable) as info:
        llm.structured("S", "U", LLMAnswer)
    assert info.value.retryable is retryable
    assert FAKE_KEY not in str(info.value)
    assert str(info.value).startswith("anthropic call failed: ")


def test_anthropic_missing_key_factory_paths(caplog: pytest.LogCaptureFixture) -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", groq_api_key=FAKE_KEY)

    with caplog.at_level(logging.WARNING, logger="studiodesk.llm.factory"):
        assert build_llm(settings) is None
    assert "llm not configured" in caplog.text
    with pytest.raises(LLMUnavailable):
        AnthropicLLM.from_settings(settings)


def test_anthropic_from_settings_pins_host_despite_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example.com")
    rec = Recorder([a_ok(anthropic_message())])
    seen = _patch_sdk(
        monkeypatch,
        anthropic_module.anthropic,
        "Anthropic",
        lambda: httpx2.Client(transport=httpx2.MockTransport(rec)),
    )
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_api_key=FAKE_KEY,
        llm_refusal_fallback=False,
        llm_effort="low",
    )

    llm = build_llm(settings)
    assert isinstance(llm, AnthropicLLM)
    llm.structured("S", "U", LLMAnswer)
    llm.close()

    assert seen[0]["max_retries"] == 0
    assert rec.requests[0].url.host == "api.anthropic.com"
    body = rec.bodies()[0]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["output_config"]["effort"] == "low"
    assert "fallbacks" not in body


def test_build_llm_without_any_key_returns_none() -> None:
    assert build_llm(Settings(_env_file=None)) is None


# --------------------------------------------------------------------------- base policy


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, None),
        ({"retry-after": "7"}, 7.0),
        ({"retry-after": "-3"}, 0.0),
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after-ms": "bad", "retry-after": "2"}, 2.0),
        ({"retry-after": "not a date"}, None),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, 0.0),  # in the past
    ],
)
def test_parse_retry_after(headers: dict[str, str], expected: float | None) -> None:
    assert parse_retry_after(headers) == expected


def test_call_with_retry_backoff_and_limits() -> None:
    sleeps: list[float] = []
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise LLMUnavailable("x", retryable=True)
        return "done"

    assert call_with_retry(flaky, max_retries=2, max_wait_s=10, sleep=sleeps.append) == "done"
    assert sleeps == [0.5, 1.0]


def test_call_with_retry_does_not_retry_non_retryable_or_zero_retries() -> None:
    attempts: list[int] = []

    def fail() -> None:
        attempts.append(1)
        raise LLMUnavailable("x", retryable=False)

    with pytest.raises(LLMUnavailable):
        call_with_retry(fail, max_retries=3, max_wait_s=10, sleep=lambda s: None)
    assert len(attempts) == 1

    def fail_retryable() -> None:
        attempts.append(1)
        raise LLMUnavailable("x", retryable=True)

    with pytest.raises(LLMUnavailable):
        call_with_retry(fail_retryable, max_retries=0, max_wait_s=10, sleep=lambda s: None)
    assert len(attempts) == 2
