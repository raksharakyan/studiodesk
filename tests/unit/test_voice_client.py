"""ElevenLabsClient on an httpx MockTransport: pinned host, auth header, URL validation,
error mapping and tool-list pagination. No network, fake key only."""

from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from studiodesk.voice.client import (
    DETAIL_MAX_CHARS,
    ELEVENLABS_API_BASE,
    MAX_LIST_PAGES,
    ElevenLabsClient,
    ElevenLabsError,
    build_http_client,
    error_detail,
    require_id,
    validate_signed_url,
)

API_KEY = "xi-unit-key-not-real-0123456789"
AGENT_ID = "agent_unit123"
GOOD_URL = "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=a&conversation_signature=s"

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler) -> tuple[ElevenLabsClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http = httpx.Client(transport=httpx.MockTransport(record))
    return ElevenLabsClient(http, SecretStr(API_KEY), timeout_s=3), seen


# --- build_http_client ---------------------------------------------------------------


def test_build_http_client_disables_redirects_and_env() -> None:
    http = build_http_client(7.5)
    try:
        assert http.follow_redirects is False
        assert http.trust_env is False
        assert http.timeout.read == 7.5
    finally:
        http.close()


# --- validate_signed_url -------------------------------------------------------------


def test_validate_signed_url_accepts_elevenlabs_wss() -> None:
    assert validate_signed_url(GOOD_URL) == GOOD_URL


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        "",
        "ws://api.elevenlabs.io/v1/convai/conversation?x=1",
        "https://api.elevenlabs.io/v1/convai/conversation?x=1",
        "wss://evil.example/v1/convai/conversation",
        "wss://api.elevenlabs.io.evil.example/v1/convai/conversation",
        "wss://evil.example#api.elevenlabs.io/v1/convai/",
        "wss://user:pw@api.elevenlabs.io/v1/convai/conversation",
        "wss://api.elevenlabs.io@evil.example/v1/convai/conversation",
        "wss://api.elevenlabs.io:8443/v1/convai/conversation",
        "wss://API.ELEVENLABS.IO/v1/convai/conversation",
        "wss://api.elevenlabs.io/v1/other/conversation",
        "wss://api.elevenlabs.io/v1/convai/conversation#frag",
        "wss://api.elevenlabs.io/v1/convai/conver sation",
        "wss://api.elevenlabs.io/v1/convai/conversation\n",
        "javascript:alert(1)",
        "wss://api.elevenlabs.io/v1/convai/" + "a" * 5000,
    ],
)
def test_validate_signed_url_rejects_bad_values_without_echoing(value: object) -> None:
    with pytest.raises(ElevenLabsError) as info:
        validate_signed_url(value)
    if isinstance(value, str) and value:
        assert value not in str(info.value)


# --- require_id ----------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "../agents", "a/b", "a?b=c", "a b", "x" * 129, None, 5])
def test_require_id_rejects_unsafe_ids(value: object) -> None:
    with pytest.raises(ElevenLabsError):
        require_id(value, "agent id")


def test_get_signed_url_rejects_unsafe_agent_id_before_any_request() -> None:
    client, seen = make_client(lambda r: httpx.Response(200, json={"signed_url": GOOD_URL}))
    with pytest.raises(ElevenLabsError):
        client.get_signed_url("agent&x=1")
    assert seen == []


# --- get_signed_url ------------------------------------------------------------------


def test_get_signed_url_request_shape() -> None:
    client, seen = make_client(lambda r: httpx.Response(200, json={"signed_url": GOOD_URL}))
    assert client.get_signed_url(AGENT_ID) == GOOD_URL
    [request] = seen
    assert request.method == "GET"
    assert request.url.scheme == "https"
    assert request.url.host == "api.elevenlabs.io"
    assert request.url.port is None
    assert request.url.path == "/v1/convai/conversation/get-signed-url"
    assert dict(request.url.params) == {"agent_id": AGENT_ID}
    assert request.headers["xi-api-key"] == API_KEY
    assert "authorization" not in request.headers
    assert request.extensions["timeout"]["read"] == 3


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"detail": "bad key"}),
        httpx.Response(404, json={"detail": {"status": "agent_not_found"}}),
        httpx.Response(429, text="slow down"),
        httpx.Response(500, text="boom"),
        httpx.Response(302, headers={"Location": "https://evil.example/steal"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["a list"]),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"signed_url": None}),
        httpx.Response(200, json={"signed_url": "wss://evil.example/v1/convai/x"}),
    ],
)
def test_get_signed_url_upstream_failures_raise(response: httpx.Response) -> None:
    client, seen = make_client(lambda r: response)
    with pytest.raises(ElevenLabsError) as info:
        client.get_signed_url(AGENT_ID)
    assert API_KEY not in str(info.value)
    assert API_KEY not in (info.value.detail or "")
    assert len(seen) == 1  # a redirect is never followed


@pytest.mark.parametrize(
    "exc",
    [httpx.ReadTimeout("t"), httpx.ConnectError("c"), httpx.RemoteProtocolError("p")],
)
def test_transport_errors_become_safe_elevenlabs_error(exc: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    client, _ = make_client(handler)
    with pytest.raises(ElevenLabsError) as info:
        client.get_signed_url(AGENT_ID)
    assert str(info.value) == f"elevenlabs request failed: {type(exc).__name__}"
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_status_error_carries_status_and_redacted_detail() -> None:
    body = {"detail": {"message": f"key {API_KEY} lacks permission\n\tconvai_write"}}
    client, _ = make_client(lambda r: httpx.Response(403, json=body))
    with pytest.raises(ElevenLabsError) as info:
        client.list_tools()
    assert str(info.value) == "elevenlabs returned status 403"
    assert info.value.status == 403
    assert info.value.detail == "key [redacted] lacks permission convai_write"


# --- error_detail --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"detail": "plain"}, "plain"),
        ({"detail": {"message": "msg"}}, "msg"),
        ({"detail": {"status": "st"}}, "st"),
        ({"detail": [{"msg": "a"}, {"msg": "b"}, "x", {"msg": "c"}, {"msg": "d"}]}, "a; b"),
        ({"detail": 5}, None),
        ({"detail": ""}, None),
        ({"other": "x"}, None),
        (["detail"], None),
    ],
)
def test_error_detail_shapes(body: object, expected: str | None) -> None:
    assert error_detail(httpx.Response(400, json=body)) == expected


def test_error_detail_ignores_non_json_and_truncates() -> None:
    assert error_detail(httpx.Response(400, text="<html>")) is None
    long = error_detail(httpx.Response(400, json={"detail": "x" * 1000}))
    assert long is not None and len(long) == DETAIL_MAX_CHARS


def test_error_detail_never_reads_headers() -> None:
    response = httpx.Response(400, headers={"x-secret": API_KEY}, json={"detail": "nope"})
    assert error_detail(response, API_KEY) == "nope"


# --- tools and agents ----------------------------------------------------------------


def test_list_tools_follows_cursor_and_skips_malformed() -> None:
    pages = {
        "": {
            "tools": [
                {"id": "t1", "tool_config": {"name": "one"}},
                {"id": "bad id!", "tool_config": {"name": "skipped"}},
                "garbage",
                {"id": "t2", "name": "legacy_name"},
            ],
            "has_more": True,
            "next_cursor": "c2",
        },
        "c2": {"tools": [{"id": "t3", "tool_config": {"name": "three"}}], "has_more": False},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[request.url.params.get("cursor", "")])

    client, seen = make_client(handler)
    tools = client.list_tools()
    assert [(t.id, t.name) for t in tools] == [
        ("t1", "one"),
        ("t2", "legacy_name"),
        ("t3", "three"),
    ]
    assert [r.url.params.get("cursor") for r in seen] == [None, "c2"]


def test_list_tools_stops_on_has_more_without_cursor() -> None:
    client, seen = make_client(lambda r: httpx.Response(200, json={"tools": [], "has_more": True}))
    assert client.list_tools() == []
    assert len(seen) == 1


def test_list_tools_bounded_when_server_never_ends() -> None:
    body = {"tools": [], "has_more": True, "next_cursor": "again"}
    client, seen = make_client(lambda r: httpx.Response(200, json=body))
    with pytest.raises(ElevenLabsError, match="did not end"):
        client.list_tools()
    assert len(seen) == MAX_LIST_PAGES


def test_list_tools_rejects_non_list_payload() -> None:
    client, _ = make_client(lambda r: httpx.Response(200, json={"tools": "nope"}))
    with pytest.raises(ElevenLabsError):
        client.list_tools()


def test_create_tool_rejects_unsafe_returned_id() -> None:
    client, _ = make_client(lambda r: httpx.Response(200, json={"id": "../x"}))
    with pytest.raises(ElevenLabsError):
        client.create_tool({"name": "t"})


def test_create_agent_rejects_missing_agent_id() -> None:
    client, _ = make_client(lambda r: httpx.Response(200, json={"id": "wrong_field"}))
    with pytest.raises(ElevenLabsError):
        client.create_agent({"name": "a"})


def test_update_paths_use_pinned_host_and_validated_ids() -> None:
    client, seen = make_client(lambda r: httpx.Response(200, json={}))
    client.update_tool("tool_1", {"name": "x"})
    client.update_agent("agent_1", {"name": "y"})
    assert [(r.method, str(r.url)) for r in seen] == [
        ("PATCH", f"{ELEVENLABS_API_BASE}/v1/convai/tools/tool_1"),
        ("PATCH", f"{ELEVENLABS_API_BASE}/v1/convai/agents/agent_1"),
    ]
    with pytest.raises(ElevenLabsError):
        client.update_agent("../../v1/other", {})
    assert len(seen) == 2
