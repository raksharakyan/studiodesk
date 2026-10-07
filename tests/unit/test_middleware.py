"""BodySizeLimitMiddleware Content-Length parsing, driven directly at the ASGI level."""

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from studiodesk.middleware import BodySizeLimitMiddleware


async def _echo_app(scope: Scope, receive: Receive, send: Send) -> None:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": body})


def _call(content_length: bytes, body: bytes = b"{}", max_bytes: int = 100) -> list[Message]:
    sent: list[Message] = []
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(b"content-length", content_length)],
    }

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(BodySizeLimitMiddleware(_echo_app, max_bytes)(scope, receive, send))
    return sent


def _status(messages: list[Message]) -> int:
    return int(next(m["status"] for m in messages if m["type"] == "http.response.start"))


@pytest.mark.parametrize(
    "value",
    [
        b"\xb2",  # superscript two: str.isdigit() is True, int() raises
        b"\xb3",
        b"\xb9",
        b"1\xb2",
        "١٢".encode(),  # Arabic-Indic digits as UTF-8 bytes
        "２".encode(),  # fullwidth digit
        b"",
        b"-1",
        b"+5",
        b"1 2",
        b"0x10",
    ],
)
def test_non_ascii_or_malformed_content_length_is_400(value: bytes) -> None:
    messages = _call(value)

    assert _status(messages) == 400
    assert b"Invalid Content-Length header" in messages[-1]["body"]


@pytest.mark.parametrize(("value", "expected"), [(b"2", 200), (b" 2 ", 200), (b"101", 413)])
def test_ascii_content_length(value: bytes, expected: int) -> None:
    assert _status(_call(value)) == expected


def test_non_ascii_content_length_via_http_client_is_400_not_500(client: TestClient) -> None:
    response = client.post(
        "/search",
        content=b'{"query": "x"}',
        headers={"content-type": "application/json", "content-length": b"\xb2"},  # type: ignore[dict-item]
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Content-Length header"}


def test_non_http_scope_passes_through() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    async def noop_receive() -> Message:
        return {}

    async def noop_send(message: Message) -> None:
        return None

    asyncio.run(BodySizeLimitMiddleware(app, 1)({"type": "lifespan"}, noop_receive, noop_send))

    assert seen == ["lifespan"]
