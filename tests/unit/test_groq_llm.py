"""GroqLLM request shape and content-only parsing (httpx MockTransport, no network)."""

import json

import groq
import httpx

from studiodesk.llm import GroqLLM
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
