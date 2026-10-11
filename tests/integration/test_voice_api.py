"""GET /voice/session, GET /voice and /static over the real app with a MockTransport
ElevenLabs client. No network; Settings never read a `.env` file."""

import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from studiodesk.api.pages import ELEVENLABS_SDK_URL
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.vectorstore import QdrantStore
from studiodesk.voice import ElevenLabsClient

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "src" / "studiodesk" / "static"

API_KEY = "xi-integration-key-not-real-0123456789"
AGENT_ID = "agent_int123"
SIGNATURE = "sig-integration-not-real-abcdef"
SIGNED_URL = (
    f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={AGENT_ID}"
    f"&conversation_signature={SIGNATURE}"
)
UPSTREAM_DETAIL = {"detail": "Voice is temporarily unavailable"}
NOT_CONFIGURED = {"detail": "Voice is not configured"}

# Exact headers the voice page must carry (hard-coded here so drift in pages.py is caught;
# only the pinned SDK URL is shared with pages.py, so the version lives in one place).
EXPECTED_CSP = (
    "default-src 'none'; "
    f"script-src 'self' {ELEVENLABS_SDK_URL}; "
    "connect-src 'self' wss://api.elevenlabs.io https://api.elevenlabs.io; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "media-src 'self' blob:; "
    "worker-src 'self' blob:; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)
EXPECTED_SECURITY_HEADERS = {
    "content-security-policy": EXPECTED_CSP,
    "permissions-policy": "microphone=(self), camera=(), geolocation=()",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}

Handler = Callable[[httpx.Request], httpx.Response]


class Upstream:
    """Fake ElevenLabs: records requests and answers with `handler`."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def client(self) -> ElevenLabsClient:
        http = httpx.Client(transport=httpx.MockTransport(self))
        return ElevenLabsClient(http, SecretStr(API_KEY), timeout_s=5)


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"signed_url": SIGNED_URL})


@contextmanager
def app_client(
    store: QdrantStore,
    embedder: Embedder,
    *,
    voice: ElevenLabsClient | None,
    static_dir: Path | None = None,
    **overrides: object,
) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, app_env="test", **overrides)  # type: ignore[arg-type]
    extra = {"static_dir": static_dir} if static_dir is not None else {}
    app = create_app(settings, embedder=embedder, store=store, voice=voice, **extra)
    with TestClient(app, follow_redirects=False) as client:
        yield client


def assert_no_secret_in_records(records: list[logging.LogRecord], *secrets: str) -> None:
    """Check raw (unredacted) records: message, args, extras and exception text."""
    for record in records:
        blob = " ".join(
            [
                record.getMessage(),
                repr(record.args),
                repr({k: v for k, v in vars(record).items() if k not in {"msg", "args"}}),
                logging.Formatter().formatException(record.exc_info) if record.exc_info else "",
            ]
        )
        for secret in secrets:
            assert secret not in blob, f"secret leaked in log record {record.name}"


# --- /voice/session happy path -------------------------------------------------------


def test_session_happy_path(
    store: QdrantStore, fake_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    upstream = Upstream(ok_handler)
    with app_client(
        store, fake_embedder, voice=upstream.client(), elevenlabs_agent_id=AGENT_ID
    ) as client:
        response = client.get("/voice/session")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"signed_url": SIGNED_URL, "expires_in_s": 900}
    [request] = upstream.requests
    assert request.method == "GET"
    assert f"{request.url.scheme}://{request.url.host}" == "https://api.elevenlabs.io"
    assert request.url.port is None
    assert request.url.path == "/v1/convai/conversation/get-signed-url"
    assert dict(request.url.params) == {"agent_id": AGENT_ID}
    assert request.headers["xi-api-key"] == API_KEY
    assert request.content == b""
    assert_no_secret_in_records(caplog.records, API_KEY, SIGNED_URL, SIGNATURE)


def test_each_session_call_mints_a_fresh_url(store: QdrantStore, fake_embedder: Embedder) -> None:
    upstream = Upstream(ok_handler)
    with app_client(
        store, fake_embedder, voice=upstream.client(), elevenlabs_agent_id=AGENT_ID
    ) as client:
        assert client.get("/voice/session").status_code == 200
        assert client.get("/voice/session").status_code == 200
    assert len(upstream.requests) == 2


def test_session_is_get_only(store: QdrantStore, fake_embedder: Embedder) -> None:
    upstream = Upstream(ok_handler)
    with app_client(
        store, fake_embedder, voice=upstream.client(), elevenlabs_agent_id=AGENT_ID
    ) as client:
        assert client.post("/voice/session").status_code == 405
    assert upstream.requests == []


# --- 503 not configured --------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "with_client"),
    [
        pytest.param({}, False, id="nothing"),
        pytest.param({"elevenlabs_agent_id": AGENT_ID}, False, id="no-key"),
        pytest.param(
            {"elevenlabs_api_key": "", "elevenlabs_agent_id": AGENT_ID}, False, id="empty-key"
        ),
        pytest.param(
            {"elevenlabs_api_key": "   ", "elevenlabs_agent_id": AGENT_ID}, False, id="blank-key"
        ),
        pytest.param({}, True, id="no-agent-id"),
        pytest.param({"elevenlabs_agent_id": ""}, True, id="empty-agent-id"),
        pytest.param({"elevenlabs_agent_id": "  "}, True, id="blank-agent-id"),
    ],
)
def test_session_503_when_not_configured(
    store: QdrantStore,
    fake_embedder: Embedder,
    overrides: dict[str, object],
    with_client: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call expected")

    upstream = Upstream(handler)
    voice = upstream.client() if with_client else None
    with app_client(store, fake_embedder, voice=voice, **overrides) as client:
        response = client.get("/voice/session")
    assert response.status_code == 503
    assert response.json() == NOT_CONFIGURED
    assert upstream.requests == []


def test_settings_treat_empty_voice_values_as_unset() -> None:
    settings = Settings(_env_file=None, elevenlabs_api_key="", elevenlabs_agent_id="")
    assert settings.elevenlabs_api_key is None
    assert settings.elevenlabs_agent_id is None


@pytest.mark.parametrize("agent_id", ["agent id", "a/b", "a&x=1", "x" * 129])
def test_settings_reject_unsafe_agent_id(agent_id: str) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, elevenlabs_agent_id=agent_id)


def test_app_builds_voice_client_only_with_key(store: QdrantStore, fake_embedder: Embedder) -> None:
    keyed = Settings(_env_file=None, app_env="test", elevenlabs_api_key=API_KEY)
    app = create_app(keyed, embedder=fake_embedder, store=store)
    with TestClient(app):
        assert isinstance(app.state.voice, ElevenLabsClient)
    bare = Settings(_env_file=None, app_env="test")
    app = create_app(bare, embedder=fake_embedder, store=store)
    with TestClient(app):
        assert app.state.voice is None


# --- 502 upstream failures -----------------------------------------------------------


def _raise(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def _respond(response: httpx.Response) -> Handler:
    return lambda request: response


UPSTREAM_FAILURES = [
    pytest.param(_respond(httpx.Response(400, json={"detail": API_KEY})), id="400"),
    pytest.param(_respond(httpx.Response(401, json={"detail": "bad key"})), id="401"),
    pytest.param(_respond(httpx.Response(404, json={"detail": "no agent"})), id="404"),
    pytest.param(_respond(httpx.Response(429, text="slow")), id="upstream-429"),
    pytest.param(_respond(httpx.Response(500, text=SIGNED_URL)), id="500"),
    pytest.param(_respond(httpx.Response(503, text="down")), id="503"),
    pytest.param(_raise(httpx.ReadTimeout("read timed out")), id="read-timeout"),
    pytest.param(_raise(httpx.ConnectTimeout("connect timed out")), id="connect-timeout"),
    pytest.param(_raise(httpx.ConnectError("refused")), id="connect-error"),
    pytest.param(_respond(httpx.Response(200, text="{not json")), id="invalid-json"),
    pytest.param(_respond(httpx.Response(200, json=[SIGNED_URL])), id="json-list"),
    pytest.param(_respond(httpx.Response(200, json={})), id="missing-signed-url"),
    pytest.param(_respond(httpx.Response(200, json={"signed_url": ""})), id="empty-signed-url"),
    pytest.param(
        _respond(httpx.Response(200, json={"signed_url": SIGNED_URL.replace("wss", "ws", 1)})),
        id="ws-scheme",
    ),
    pytest.param(
        _respond(httpx.Response(200, json={"signed_url": SIGNED_URL.replace("wss", "https", 1)})),
        id="https-scheme",
    ),
    pytest.param(
        _respond(
            httpx.Response(
                200,
                json={"signed_url": SIGNED_URL.replace("api.elevenlabs.io", "evil.example")},
            )
        ),
        id="foreign-host",
    ),
    pytest.param(
        _respond(
            httpx.Response(
                200,
                json={
                    "signed_url": SIGNED_URL.replace(
                        "api.elevenlabs.io", "api.elevenlabs.io.evil.example"
                    )
                },
            )
        ),
        id="lookalike-host",
    ),
    pytest.param(
        _respond(
            httpx.Response(
                200,
                json={
                    "signed_url": SIGNED_URL.replace(
                        "api.elevenlabs.io", "api.elevenlabs.io@evil.example"
                    )
                },
            )
        ),
        id="userinfo-host",
    ),
]


@pytest.mark.parametrize("handler", UPSTREAM_FAILURES)
def test_session_502_on_upstream_failure(
    store: QdrantStore,
    fake_embedder: Embedder,
    caplog: pytest.LogCaptureFixture,
    handler: Handler,
) -> None:
    caplog.set_level(logging.DEBUG)
    upstream = Upstream(handler)
    with app_client(
        store, fake_embedder, voice=upstream.client(), elevenlabs_agent_id=AGENT_ID
    ) as client:
        response = client.get("/voice/session")

    assert response.status_code == 502
    assert response.json() == UPSTREAM_DETAIL
    assert response.headers["cache-control"] == "no-store"
    for secret in (API_KEY, SIGNATURE, "evil.example"):
        assert secret not in response.text
        assert secret not in str(response.headers)
    assert len(upstream.requests) == 1
    assert any(r.getMessage() == "voice session failed" for r in caplog.records)
    assert_no_secret_in_records(caplog.records, API_KEY, SIGNATURE, "evil.example")


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_session_never_follows_redirects(
    store: QdrantStore, fake_embedder: Embedder, status: int
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "api.elevenlabs.io":
            raise AssertionError(f"redirect followed to {request.url}")
        return httpx.Response(status, headers={"Location": "https://evil.example/collect"})

    upstream = Upstream(handler)
    with app_client(
        store, fake_embedder, voice=upstream.client(), elevenlabs_agent_id=AGENT_ID
    ) as client:
        response = client.get("/voice/session")
    assert response.status_code == 502
    assert len(upstream.requests) == 1
    assert "evil.example" not in response.text


def test_session_rate_limited(store: QdrantStore, fake_embedder: Embedder) -> None:
    upstream = Upstream(ok_handler)
    with app_client(
        store,
        fake_embedder,
        voice=upstream.client(),
        elevenlabs_agent_id=AGENT_ID,
        voice_rate_limit="2/minute",
    ) as client:
        statuses = [client.get("/voice/session").status_code for _ in range(3)]
        limited = client.get("/voice/session")
    assert statuses == [200, 200, 429]
    assert limited.status_code == 429
    assert SIGNATURE not in limited.text
    assert int(limited.headers["retry-after"]) > 0
    assert len(upstream.requests) == 2


# --- pages and static ----------------------------------------------------------------


def assert_security_headers(response: httpx.Response) -> None:
    for name, value in EXPECTED_SECURITY_HEADERS.items():
        assert response.headers.get(name) == value, name


def test_voice_page_serves_real_html_with_exact_headers(
    store: QdrantStore, fake_embedder: Embedder
) -> None:
    with app_client(store, fake_embedder, voice=None) as client:
        response = client.get("/voice")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.text == (STATIC_DIR / "voice.html").read_text(encoding="utf-8")
    assert_security_headers(response)


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/voice.js", "javascript"),
        ("/static/styles.css", "text/css"),
        ("/static/voice.html", "text/html"),
    ],
)
def test_static_assets_have_security_headers(
    store: QdrantStore, fake_embedder: Embedder, path: str, content_type: str
) -> None:
    with app_client(store, fake_embedder, voice=None) as client:
        response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert_security_headers(response)


@pytest.mark.parametrize(
    "path",
    ["/static/../config.py", "/static/%2e%2e/config.py", "/static/missing.js", "/static/"],
)
def test_static_does_not_escape_or_list(
    store: QdrantStore, fake_embedder: Embedder, path: str
) -> None:
    with app_client(store, fake_embedder, voice=None) as client:
        response = client.get(path)
    assert response.status_code == 404
    assert "Settings" not in response.text


def test_pages_404_when_static_dir_missing(
    store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    missing = tmp_path / "no-static"
    with app_client(store, fake_embedder, voice=None, static_dir=missing) as client:
        assert client.get("/voice").status_code == 404
        assert client.get("/static/voice.js").status_code == 404
        assert client.get("/health").status_code == 200


def test_voice_page_404_when_html_missing(
    store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    (tmp_path / "voice.js").write_text("export {};")
    with app_client(store, fake_embedder, voice=None, static_dir=tmp_path) as client:
        assert client.get("/voice").status_code == 404
        assert client.get("/static/voice.js").status_code == 200


def test_pages_excluded_from_openapi(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder, voice=None) as client:
        paths = client.get("/openapi.json").json()["paths"]
    assert "/voice" not in paths
    assert "/voice/session" in paths


# --- voice.html static analysis ------------------------------------------------------


class _HTMLScan(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[dict[str, str | None]] = []
        self.inline_script_bodies: list[str] = []
        self.event_attrs: list[tuple[str, str]] = []
        self.style_attrs: list[str] = []
        self.links: list[dict[str, str | None]] = []
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        for name, _ in attrs:
            if name.lower().startswith("on"):
                self.event_attrs.append((tag, name))
            if name.lower() == "style":
                self.style_attrs.append(tag)
        if tag == "script":
            self.scripts.append(attr_map)
            self._in_script = True
        if tag == "link":
            self.links.append(attr_map)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline_script_bodies.append(data)


@pytest.fixture(scope="module")
def voice_html() -> _HTMLScan:
    scan = _HTMLScan()
    scan.feed((STATIC_DIR / "voice.html").read_text(encoding="utf-8"))
    return scan


def test_voice_html_has_no_inline_script_or_handlers(voice_html: _HTMLScan) -> None:
    assert voice_html.scripts, "expected at least one <script src>"
    assert voice_html.inline_script_bodies == []
    assert all(script.get("src") for script in voice_html.scripts)
    assert voice_html.event_attrs == []
    assert voice_html.style_attrs == []  # CSP style-src 'self' blocks inline styles


def test_voice_html_scripts_use_no_javascript_urls() -> None:
    text = (STATIC_DIR / "voice.html").read_text(encoding="utf-8")
    assert not re.search(r"javascript:", text, re.IGNORECASE)


def test_external_scripts_have_sri_and_crossorigin(voice_html: _HTMLScan) -> None:
    external = [s for s in voice_html.scripts if urlsplit(s.get("src") or "").netloc]
    assert external, "the ElevenLabs SDK is expected from the CDN"
    for script in external:
        src = script["src"] or ""
        parts = urlsplit(src)
        assert parts.scheme == "https", src
        assert f"{parts.scheme}://{parts.netloc}" == "https://cdn.jsdelivr.net", src
        assert re.search(r"@\d+\.\d+\.\d+/", parts.path), f"version not pinned: {src}"
        assert src == ELEVENLABS_SDK_URL, "the CSP script-src allows only ELEVENLABS_SDK_URL"
        assert re.fullmatch(r"sha(384|512)-[A-Za-z0-9+/]{40,}={0,2}", script.get("integrity") or "")
        assert script.get("crossorigin") == "anonymous"


def test_local_scripts_and_styles_are_same_origin(voice_html: _HTMLScan) -> None:
    local = [s for s in voice_html.scripts if not urlsplit(s.get("src") or "").netloc]
    assert [s["src"] for s in local] == ["/static/voice.js"]
    for link in voice_html.links:
        href = link.get("href") or ""
        assert not urlsplit(href).netloc, f"external stylesheet/link: {href}"


def _resolve_worklet_paths(js: str) -> dict[str, str]:
    """Worklet URLs from WORKLET_PATHS in voice.js, with top-level string constants expanded."""
    block = re.search(r"const WORKLET_PATHS = Object\.freeze\(\{(.*?)\}\);", js, re.DOTALL)
    assert block is not None, "WORKLET_PATHS not found in voice.js"
    consts = dict(re.findall(r'^const ([A-Z][A-Z0-9_]*) = [`"]([^`"]*)[`"];', js, re.MULTILINE))
    paths: dict[str, str] = {}
    for name, value in re.findall(r'(\w+):\s*[`"]([^`"]+)[`"]', block.group(1)):
        for _ in range(3):
            for const, const_value in consts.items():
                value = value.replace(f"${{{const}}}", const_value)
        paths[name] = value
    return paths


def test_worklets_are_vendored_or_pinned_to_the_sdk_version() -> None:
    """Tolerant until the worklets are vendored: same-origin /static/vendor/ files that
    exist, or CDN URLs pinned to the same SDK version as the <script> tag in voice.html."""
    html = (STATIC_DIR / "voice.html").read_text(encoding="utf-8")
    js = (STATIC_DIR / "voice.js").read_text(encoding="utf-8")
    paths = _resolve_worklet_paths(js)
    assert set(paths) == {"rawAudioProcessor", "audioConcatProcessor"}
    sdk_versions = re.findall(r"@elevenlabs/client@(\d+\.\d+\.\d+)/", html)
    for name, url in paths.items():
        assert "${" not in url, f"unresolved template in {name}: {url}"
        parts = urlsplit(url)
        if not parts.netloc:
            assert url.startswith("/static/vendor/"), f"{name}: {url}"
            assert ".." not in url
            vendored = STATIC_DIR / url.removeprefix("/static/")
            assert vendored.is_file(), f"{name}: vendored worklet missing at {vendored}"
            # The vendored copy must come from the same SDK version as the <script> tag.
            dir_versions = re.findall(r"elevenlabs-client-(\d+\.\d+\.\d+)/", url)
            assert dir_versions == sdk_versions, f"{name} vendored from another SDK version"
        else:
            assert f"{parts.scheme}://{parts.netloc}" == "https://cdn.jsdelivr.net", url
            [url_version] = re.findall(r"@elevenlabs/client@(\d+\.\d+\.\d+)/", url)
            assert sdk_versions == [url_version], f"{name} not pinned to the SDK version"


def test_worklet_parser_handles_vendored_form() -> None:
    js = (
        "const WORKLET_PATHS = Object.freeze({\n"
        '  rawAudioProcessor: "/static/vendor/elevenlabs/rawAudioProcessor.js",\n'
        "  audioConcatProcessor: `/static/vendor/elevenlabs/audioConcatProcessor.js`,\n"
        "});\n"
    )
    assert _resolve_worklet_paths(js) == {
        "rawAudioProcessor": "/static/vendor/elevenlabs/rawAudioProcessor.js",
        "audioConcatProcessor": "/static/vendor/elevenlabs/audioConcatProcessor.js",
    }


def test_vendored_worklets_served_same_origin_with_headers(
    store: QdrantStore, fake_embedder: Embedder
) -> None:
    js = (STATIC_DIR / "voice.js").read_text(encoding="utf-8")
    local = [u for u in _resolve_worklet_paths(js).values() if u.startswith("/static/vendor/")]
    if not local:
        pytest.skip("worklets not vendored yet (still loaded from the pinned CDN)")
    with app_client(store, fake_embedder, voice=None) as client:
        for url in local:
            response = client.get(url)
            assert response.status_code == 200, url
            assert "javascript" in response.headers["content-type"], url
            assert_security_headers(response)
