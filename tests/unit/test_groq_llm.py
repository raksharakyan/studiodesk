"""GroqLLM request shape and content-only parsing (httpx MockTransport, no network)."""

import json

import groq
import httpx
import pytest

from studiodesk.llm import GroqLLM, LLMUnavailable
from studiodesk.models.answer import LLMAnswer

VALID = '{"answer": "ok", "cited_ids": ["BUG-0001"], "insufficient_context": false}'


def test_sends_reasoning_params_and_parses_only_content() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        message = {
            "role": "assistant",
            "content": VALID,
            # Must never be parsed as the answer.
            "reasoning": '{"answer": "LEAKED", "cited_ids": [], "insufficient_context": true}',
        }
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": message, "logprobs": None}
                ],
            },
        )

    client = groq.Groq(
        api_key="test-key-not-real",
        base_url="https://api.groq.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    llm = GroqLLM(client, model="openai/gpt-oss-120b", max_tokens=100, reasoning_effort="low")

    result = llm.structured("SYSTEM", "USER", LLMAnswer)

    assert result.answer == "ok"
    body = bodies[0]
    assert body["reasoning_effort"] == "low"
    assert body["include_reasoning"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 100


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
            }
        ],
    }


def _llm(responses: list[httpx.Response], sleeps: list[float]) -> tuple[GroqLLM, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return responses[len(calls) - 1]

    client = groq.Groq(
        api_key="test-key-not-real",
        base_url="https://api.groq.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    llm = GroqLLM(
        client,
        model="openai/gpt-oss-120b",
        max_tokens=100,
        max_retries=1,
        max_retry_wait_s=10,
        sleep=sleeps.append,
    )
    return llm, calls


def test_short_retry_after_is_honoured_once() -> None:
    sleeps: list[float] = []
    llm, calls = _llm(
        [
            httpx.Response(429, headers={"retry-after": "2"}, json={"error": {}}),
            httpx.Response(200, json=_completion(VALID)),
        ],
        sleeps,
    )

    assert llm.structured("S", "U", LLMAnswer).answer == "ok"
    assert sleeps == [2.0]
    assert len(calls) == 2


def test_long_retry_after_fails_fast_without_sleeping() -> None:
    sleeps: list[float] = []
    llm, calls = _llm(
        [httpx.Response(429, headers={"retry-after": "30"}, json={"error": {}})], sleeps
    )

    with pytest.raises(LLMUnavailable) as info:
        llm.structured("S", "U", LLMAnswer)

    assert info.value.retry_after_s == 30.0
    assert sleeps == []
    assert len(calls) == 1
