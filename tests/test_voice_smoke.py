"""Minimal smoke tests for GET /voice/session and the /voice page (MockTransport, no
network). The full suite is written by qa-agent."""

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.vectorstore import QdrantStore
from studiodesk.voice import ElevenLabsClient

API_KEY = "xi-test-key-not-real"
AGENT_ID = "agent_test123"
SIGNED_URL = (
    f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={AGENT_ID}"
    "&conversation_signature=sig-not-real"
)


def voice_client(handler: httpx.MockTransport) -> ElevenLabsClient:
    return ElevenLabsClient(httpx.Client(transport=handler), SecretStr(API_KEY), timeout_s=5)


def make_client(
    store: QdrantStore,
    embedder: Embedder,
    voice: ElevenLabsClient | None,
    static_dir: Path | None = None,
    **overrides: object,
) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, app_env="test", **overrides)  # type: ignore[arg-type]
    extra = {"static_dir": static_dir} if static_dir is not None else {}
    app = create_app(settings, embedder=embedder, store=store, voice=voice, **extra)
    with TestClient(app) as client:
        yield client


def test_voice_session_returns_signed_url_uncached(
    store: QdrantStore, fake_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"signed_url": SIGNED_URL})

    voice = voice_client(httpx.MockTransport(handler))
    for client in make_client(store, fake_embedder, voice, elevenlabs_agent_id=AGENT_ID):
        response = client.get("/voice/session")

    assert response.status_code == 200
    assert response.json() == {"signed_url": SIGNED_URL, "expires_in_s": 900}
    assert response.headers["Cache-Control"] == "no-store"
    [request] = requests
    assert request.url.host == "api.elevenlabs.io"
    assert request.url.path == "/v1/convai/conversation/get-signed-url"
    assert request.url.params["agent_id"] == AGENT_ID
    assert request.headers["xi-api-key"] == API_KEY
    assert "conversation_signature" not in caplog.text


def test_voice_session_rejects_foreign_signed_url(
    store: QdrantStore, fake_embedder: Embedder
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"signed_url": "wss://evil.example/v1/convai/x"})

    voice = voice_client(httpx.MockTransport(handler))
    for client in make_client(store, fake_embedder, voice, elevenlabs_agent_id=AGENT_ID):
        response = client.get("/voice/session")

    assert response.status_code == 502
    assert response.json() == {"detail": "Voice is temporarily unavailable"}


@pytest.mark.parametrize("with_client", [False, True])
def test_voice_session_503_when_not_configured(
    store: QdrantStore, fake_embedder: Embedder, with_client: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call expected")

    # Without a key no client is built; with a client but no agent id it is still 503.
    voice = voice_client(httpx.MockTransport(handler)) if with_client else None
    for client in make_client(store, fake_embedder, voice):
        response = client.get("/voice/session")

    assert response.status_code == 503
    assert response.json() == {"detail": "Voice is not configured"}


def test_voice_page_has_security_headers(
    store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    (tmp_path / "voice.html").write_text("<!doctype html><title>Voice</title>")
    for client in make_client(store, fake_embedder, None, static_dir=tmp_path):
        page = client.get("/voice")
        asset = client.get("/static/voice.html")

    for response in (page, asset):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        csp = response.headers["Content-Security-Policy"]
        assert "script-src 'self' https://cdn.jsdelivr.net;" in csp
        assert "connect-src 'self' wss://api.elevenlabs.io https://api.elevenlabs.io;" in csp
        assert response.headers["Permissions-Policy"].startswith("microphone=(self)")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "no-referrer"
