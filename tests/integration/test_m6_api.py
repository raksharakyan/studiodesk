"""M6 backend over the real app: security headers middleware, the input-free 422 handler,
GET /api/demo-info, "file anyway", the pages and /ask source metadata.

In-memory Qdrant, fake embedder/LLM, fake GitHub and ElevenLabs over MockTransport, fixed
clocks for day rollover. No network; Settings never read a `.env` file.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import SecretStr

from fakes import FAKE_REPO, FakeGitHubAPI, FakeLLM
from studiodesk.actions.proposals import InMemoryProposalStore
from studiodesk.api.demo import build_demo_info
from studiodesk.api.errors import MAX_LOC_PART_CHARS, validation_exception_handler
from studiodesk.api.pages import CONTENT_SECURITY_POLICY, STATIC_DIR
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.models.answer import LLMAnswer
from studiodesk.models.bugs import CandidateJudgement, DuplicateJudgements
from studiodesk.vectorstore import QdrantStore
from studiodesk.voice import ElevenLabsClient, VoiceSessionQuota

QDRANT_URL = "https://example.cloud.qdrant.io"
AGENT_ID = "agent_m6secret123"
XI_KEY = "xi-m6-key-not-real-0123456789"
SIGNED_URL = "wss://api.elevenlabs.io/v1/convai/conversation?conversation_signature=sig"

PAGE_HEADERS = {
    "content-security-policy": CONTENT_SECURITY_POLICY,
    "permissions-policy": "microphone=(self), camera=(), geolocation=()",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
}
BASELINE = {"x-content-type-options": "nosniff", "referrer-policy": "no-referrer"}
PROD_ONLY = {
    "strict-transport-security": "max-age=31536000; includeSubDomains",
    "cross-origin-opener-policy": "same-origin",
}

# A paraphrase of BUG-0002 (O2 loss after host migration); BUG-0006 and BUG-0011 are
# labelled duplicates of BUG-0002, so their canonical id is BUG-0002.
O2_REPORT = {
    "title": "Oxygen drains for all clients after the host leaves co-op",
    "description": "When the host quits and host migration happens, everyone except the new "
    "host suffocates although the reactor shows 100 percent. We all died of no air.",
    "platform": "pc",
    "version": "1.0.2",
}
NEW_REPORT = {
    "title": "Photo mode camera ignores inverted Y axis",
    "description": "Invert Y works in gameplay but the photo mode free camera ignores it.",
    "platform": "ps5",
    "version": "1.4.2",
}


def judge(answer: bool) -> Callable[[str, str], DuplicateJudgements]:
    """Responder that judges every candidate in the prompt `answer`."""

    def respond(system: str, user: str) -> DuplicateJudgements:
        ids = [part.split('"')[0] for part in user.split('<document id="')[1:]]
        return DuplicateJudgements(
            judgements=[
                CandidateJudgement(candidate_id=i, is_duplicate=answer, confidence=0.9, reason="r")
                for i in ids
            ]
        )

    return respond


VERDICT_SETTINGS: dict[str, tuple[bool, dict[str, float]]] = {
    "duplicate": (True, {"dup_candidate_threshold": 0.0, "dup_auto_threshold": 0.0}),
    "possible_duplicate": (True, {"dup_candidate_threshold": 0.0, "dup_auto_threshold": 1.0}),
    "new": (False, {"dup_candidate_threshold": 1.0, "dup_auto_threshold": 1.0}),
}


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def voice_client(handler: Callable[[httpx.Request], httpx.Response]) -> ElevenLabsClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return ElevenLabsClient(http, SecretStr(XI_KEY), timeout_s=5)


def ok_voice(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"signed_url": SIGNED_URL})


@contextmanager
def app_client(
    store: QdrantStore,
    embedder: Embedder,
    *,
    llm: FakeLLM | None = None,
    github_api: FakeGitHubAPI | None = None,
    voice: ElevenLabsClient | None = None,
    proposals: InMemoryProposalStore | None = None,
    voice_quota: VoiceSessionQuota | None = None,
    static_dir: Path | None = None,
    **overrides: object,
) -> Iterator[TestClient]:
    fields: dict[str, object] = {"app_env": "test", **overrides}
    if github_api is not None:
        fields.setdefault("github_repo", FAKE_REPO)
    settings = Settings(_env_file=None, **fields)  # type: ignore[arg-type]
    extra: dict[str, Any] = {"static_dir": static_dir} if static_dir is not None else {}
    app = create_app(
        settings,
        embedder=embedder,
        store=store,
        llm=llm,
        github=github_api.client() if github_api else None,
        voice=voice,
        proposals=proposals,
        voice_quota=voice_quota,
        **extra,
    )
    with TestClient(app, follow_redirects=False) as client:
        yield client


def verdict_client(store: QdrantStore, embedder: Embedder, verdict: str, **kwargs: Any) -> Any:
    answer, thresholds = VERDICT_SETTINGS[verdict]
    llm = FakeLLM({DuplicateJudgements: judge(answer)})  # type: ignore[dict-item]
    return app_client(store, embedder, llm=llm, **{**thresholds, **kwargs})


# =====================================================================================
# Security headers middleware
# =====================================================================================


def assert_baseline(response: httpx.Response, *, prod: bool) -> None:
    for name, value in BASELINE.items():
        assert response.headers.get_list(name) == [value], name
    for name, value in PROD_ONLY.items():
        if prod:
            assert response.headers.get_list(name) == [value], name
        else:
            assert name not in response.headers, name


@pytest.mark.parametrize("prod", [False, True])
def test_headers_on_error_and_static_responses(
    store: QdrantStore, fake_embedder: Embedder, prod: bool
) -> None:
    env: dict[str, object] = (
        {"app_env": "prod", "qdrant_url": QDRANT_URL} if prod else {"app_env": "test"}
    )
    with app_client(
        store,
        fake_embedder,
        max_request_bytes=200,
        search_rate_limit="1/minute",
        **env,
    ) as client:
        responses = {
            "404": client.get("/definitely-not-here"),
            "405": client.put("/health"),
            "413": client.post(
                "/search",
                content=b'{"query": "' + b"x" * 500 + b'"}',
                headers={"content-type": "application/json"},
            ),
            "422": client.post("/search", json={}),
            "200-demo": client.get("/api/demo-info"),
            "429": client.get("/api/demo-info"),
            "503-voice": client.get("/voice/session"),
            "static-js": client.get("/static/voice.js"),
            "static-404": client.get("/static/missing.css"),
            "page": client.get("/"),
        }
    assert responses["413"].status_code == 413
    assert responses["429"].status_code == 429
    assert responses["static-js"].status_code == 200
    for name, response in responses.items():
        try:
            assert_baseline(response, prod=prod)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None


def test_page_headers_are_not_duplicated_or_overridden(
    store: QdrantStore, fake_embedder: Embedder
) -> None:
    with app_client(store, fake_embedder, app_env="prod", qdrant_url=QDRANT_URL) as client:
        for path in ("/", "/voice", "/static/home.js", "/static/index.html"):
            response = client.get(path)
            assert response.status_code == 200, path
            for name, value in PAGE_HEADERS.items():
                assert response.headers.get_list(name) == [value], (path, name)
            for name, value in PROD_ONLY.items():
                assert response.headers.get_list(name) == [value], (path, name)


def test_json_api_responses_carry_no_page_csp(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder) as client:
        response = client.get("/health")
    assert "content-security-policy" not in response.headers


# =====================================================================================
# 422 handler
# =====================================================================================

MARKER = "INPUT-MARKER-7f3a"


def assert_input_free(response: httpx.Response) -> list[dict[str, Any]]:
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    for error in detail:
        assert set(error) == {"loc", "type", "msg"}
        assert error["msg"] == "Invalid value"
        assert all(isinstance(part, str | int) for part in error["loc"])
    for leak in ("input", "ctx", "url", "max_length", "Input should", "String should"):
        assert leak not in response.text, leak
    return detail  # type: ignore[no-any-return]


def test_422_nested_location(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder) as client:
        response = client.post(
            "/search", json={"query": "save bug", "filters": {"platforms": [MARKER]}}
        )
    detail = assert_input_free(response)
    assert ["body", "filters", "platforms", 0] in [error["loc"] for error in detail]
    assert MARKER not in response.text


def test_422_long_unexpected_key_is_truncated(store: QdrantStore, fake_embedder: Embedder) -> None:
    key = MARKER + "k" * 300
    with app_client(store, fake_embedder) as client:
        response = client.post("/search", json={"query": "save bug", key: "v"})
    detail = assert_input_free(response)
    [error] = detail
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == ["body", key[:MAX_LOC_PART_CHARS]]
    assert len(error["loc"][1]) == MAX_LOC_PART_CHARS == 64
    assert key not in response.text


@pytest.mark.parametrize(
    "content",
    [b"{not json", b'{"query": "x"', b"\xff\xfe", b'{"query": "' + MARKER.encode() + b'",}'],
)
def test_422_json_decode_errors(
    store: QdrantStore, fake_embedder: Embedder, content: bytes
) -> None:
    with app_client(store, fake_embedder) as client:
        response = client.post(
            "/search", content=content, headers={"content-type": "application/json"}
        )
    detail = assert_input_free(response)
    assert detail[0]["type"] == "json_invalid"
    assert detail[0]["loc"][0] == "body"
    assert MARKER not in response.text


def test_422_long_string_value_never_echoed_or_logged(
    store: QdrantStore, fake_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    # A fake LLM: without one, /ask answers 503 from its dependency before validating.
    with app_client(store, fake_embedder, llm=FakeLLM({})) as client:
        response = client.post("/ask", json={"question": MARKER * 500})
    assert_input_free(response)
    assert MARKER not in response.text
    for record in caplog.records:
        assert MARKER not in record.getMessage()
        assert MARKER not in repr(vars(record))


def test_422_on_every_json_route(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder, llm=FakeLLM({}), github_api=FakeGitHubAPI()) as client:
        for path in ("/search", "/ask", "/bugs/check", "/actions/abc/confirm"):
            response = client.post(path, json={MARKER: MARKER})
            assert_input_free(response)
            assert MARKER not in json.dumps(
                [e["loc"] for e in response.json()["detail"] if e["type"] != "extra_forbidden"]
            )


def test_handler_with_a_foreign_exception_returns_empty_detail() -> None:
    response = asyncio.run(validation_exception_handler(None, ValueError(MARKER)))  # type: ignore[arg-type]
    assert response.status_code == 422
    assert json.loads(response.body) == {"detail": []}


def test_handler_drops_input_ctx_and_custom_msg() -> None:
    exc = RequestValidationError(
        [
            {
                "loc": ("body", "x" * 100, 3),
                "type": "string_too_long",
                "msg": f"secret {MARKER}",
                "input": MARKER,
                "ctx": {"max_length": 5, "echo": MARKER},
            },
            {"msg": "no loc or type"},
        ]
    )
    response = asyncio.run(validation_exception_handler(None, exc))  # type: ignore[arg-type]
    assert json.loads(response.body) == {
        "detail": [
            {"loc": ["body", "x" * 64, 3], "type": "string_too_long", "msg": "Invalid value"},
            {"loc": [], "type": "value_error", "msg": "Invalid value"},
        ]
    }


# =====================================================================================
# GET /api/demo-info
# =====================================================================================


def demo(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/demo-info")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body: dict[str, Any] = response.json()
    for feature in ("actions", "voice"):
        for key in ("max_per_day", "remaining_today"):
            value = body[feature][key]
            assert isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return body


def file_new_issue(client: TestClient) -> httpx.Response:
    body = client.post("/bugs/check", json=NEW_REPORT).json()
    action = body["proposed_action"]
    assert action["confirm_token"], body
    return client.post(
        f"/actions/{action['action_id']}/confirm", json={"confirm_token": action["confirm_token"]}
    )


def test_demo_info_counts_down_after_confirm(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    github_api = FakeGitHubAPI()
    with verdict_client(
        ingested_store, fake_embedder, "new", github_api=github_api, actions_max_per_day=3
    ) as client:
        assert demo(client)["actions"]["remaining_today"] == 3
        client.post("/bugs/check", json=NEW_REPORT)  # a pending proposal takes no slot
        assert demo(client)["actions"]["remaining_today"] == 3
        assert file_new_issue(client).status_code == 200
        assert demo(client)["actions"]["remaining_today"] == 2
        assert file_new_issue(client).status_code == 200
        assert file_new_issue(client).status_code == 200
        info = demo(client)
    assert info["actions"] == {
        "enabled": True,
        "configured": True,
        "max_per_day": 3,
        "remaining_today": 0,
    }
    assert len(github_api.requests) == 3


def test_demo_info_slot_released_when_filing_fails(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    github_api = FakeGitHubAPI(status_code=500)
    with verdict_client(
        ingested_store, fake_embedder, "new", github_api=github_api, actions_max_per_day=2
    ) as client:
        assert file_new_issue(client).status_code == 502
        assert demo(client)["actions"]["remaining_today"] == 2
    assert len(github_api.requests) == 1


def test_demo_info_voice_counts_reserve_and_release(
    store: QdrantStore, fake_embedder: Embedder
) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500) if calls["n"] == 2 else ok_voice(request)

    with app_client(
        store,
        fake_embedder,
        voice=voice_client(handler),
        elevenlabs_agent_id=AGENT_ID,
        voice_max_sessions_per_day=3,
    ) as client:
        assert demo(client)["voice"]["remaining_today"] == 3
        assert client.get("/voice/session").status_code == 200
        assert demo(client)["voice"]["remaining_today"] == 2
        assert client.get("/voice/session").status_code == 502  # released
        assert demo(client)["voice"]["remaining_today"] == 2
        assert client.get("/voice/session").status_code == 200
        assert client.get("/voice/session").status_code == 200
        assert demo(client)["voice"]["remaining_today"] == 0
        assert client.get("/voice/session").status_code == 503
        assert demo(client)["voice"]["remaining_today"] == 0


def test_demo_info_never_reserves(store: QdrantStore, fake_embedder: Embedder) -> None:
    quota = VoiceSessionQuota(2)
    with app_client(
        store,
        fake_embedder,
        voice=voice_client(ok_voice),
        voice_quota=quota,
        elevenlabs_agent_id=AGENT_ID,
        voice_max_sessions_per_day=2,
    ) as client:
        for _ in range(5):
            demo(client)
    assert quota.used_today() == 0


def test_demo_info_utc_rollover(ingested_store: QdrantStore, fake_embedder: Embedder) -> None:
    clock = Clock(datetime(2026, 10, 16, 23, 59, 0, tzinfo=UTC))
    proposals = InMemoryProposalStore(900, max_pending=10, max_per_day=1, clock=clock)
    quota = VoiceSessionQuota(1, clock=clock)
    with verdict_client(
        ingested_store,
        fake_embedder,
        "new",
        github_api=FakeGitHubAPI(),
        proposals=proposals,
        voice_quota=quota,
        voice=voice_client(ok_voice),
        elevenlabs_agent_id=AGENT_ID,
        actions_max_per_day=1,
        voice_max_sessions_per_day=1,
    ) as client:
        assert file_new_issue(client).status_code == 200
        assert client.get("/voice/session").status_code == 200
        info = demo(client)
        assert (info["actions"]["remaining_today"], info["voice"]["remaining_today"]) == (0, 0)
        clock.now += timedelta(minutes=2)  # 00:01 UTC next day
        info = demo(client)
    assert (info["actions"]["remaining_today"], info["voice"]["remaining_today"]) == (1, 1)


def test_demo_info_rate_limited(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder, search_rate_limit="2/minute") as client:
        statuses = [client.get("/api/demo-info").status_code for _ in range(3)]
        limited = client.get("/api/demo-info")
    assert statuses == [200, 200, 429]
    assert int(limited.headers["retry-after"]) > 0


def test_demo_info_has_no_secrets_or_agent_id(store: QdrantStore, fake_embedder: Embedder) -> None:
    secrets = {
        "groq_api_key": "gsk_m6-secret-groq-0123456789",
        "anthropic_api_key": "sk-ant-m6-secret-0123456789",
        "github_token": "github_pat_m6secret0123456789",
        "slack_webhook_url": "https://hooks.slack.com/services/T000/B000/m6secretwebhook",
        "qdrant_api_key": "qdrant-m6-secret-0123456789",
        "elevenlabs_api_key": XI_KEY,
        "elevenlabs_agent_id": AGENT_ID,
    }
    with app_client(
        store,
        fake_embedder,
        github_api=FakeGitHubAPI(),
        voice=voice_client(ok_voice),
        **secrets,
    ) as client:
        response = client.get("/api/demo-info")
    text = response.text
    for value in [*secrets.values(), "test-token-not-real", "wss://", "signature"]:
        assert value not in text, value
    # Known public fields (new non-secret fields may be added; the values check above guards
    # against leaks).
    assert {
        "game",
        "source_repo_url",
        "sandbox_repo",
        "sandbox_issues_url",
        "actions",
        "voice",
    } <= set(response.json())


def test_demo_info_prod_defaults(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder, app_env="prod", qdrant_url=QDRANT_URL) as client:
        info = demo(client)
    assert info["actions"] == {
        "enabled": False,
        "configured": False,
        "max_per_day": 20,
        "remaining_today": 20,
    }
    assert info["voice"] == {
        "enabled": False,
        "configured": False,
        "max_per_day": 30,
        "remaining_today": 30,
    }
    assert info["source_repo_url"] == "https://github.com/raksharakyan/studiodesk"


def test_demo_info_voice_needs_key_and_agent_id(
    store: QdrantStore, fake_embedder: Embedder
) -> None:
    with app_client(store, fake_embedder, voice=voice_client(ok_voice)) as client:
        assert demo(client)["voice"]["configured"] is False
    with app_client(store, fake_embedder, elevenlabs_agent_id=AGENT_ID) as client:
        assert demo(client)["voice"]["configured"] is False


@pytest.mark.parametrize(
    ("store_cap", "settings_cap", "expected"),
    [(None, 5, 5), (10, 3, 3), (2, 5, 2)],
)
def test_demo_info_remaining_never_exceeds_cap(
    store: QdrantStore,
    fake_embedder: Embedder,
    store_cap: int | None,
    settings_cap: int,
    expected: int,
) -> None:
    proposals = InMemoryProposalStore(900, max_pending=10, max_per_day=store_cap)
    with app_client(
        store, fake_embedder, proposals=proposals, actions_max_per_day=settings_cap
    ) as client:
        info = demo(client)
    assert info["actions"]["max_per_day"] == settings_cap
    assert info["actions"]["remaining_today"] == expected


def test_build_demo_info_without_collaborators_is_zero() -> None:
    settings = Settings(_env_file=None, app_env="test")
    info = build_demo_info(settings, None, None, False, None)
    assert info.actions.remaining_today == 0
    assert info.voice.remaining_today == 0
    assert info.sandbox_repo is None and info.sandbox_issues_url is None


def test_source_repo_url_setting_validated() -> None:
    ok = Settings(_env_file=None, source_repo_url="https://github.com/o/r")
    assert ok.source_repo_url == "https://github.com/o/r"
    for bad in (
        "http://github.com/o/r",
        "https://user:pw@github.com/o/r",
        "https://github.com:8443/o/r",
        "https://github.com/o/r?x=1",
        "https://github.com/o/r#frag",
        "javascript:alert(1)",
        "https://github.com/<script>",
    ):
        with pytest.raises(ValueError):
            Settings(_env_file=None, source_repo_url=bad)


# =====================================================================================
# file_anyway
# =====================================================================================


def check(client: TestClient, report: dict[str, Any], **extra: Any) -> dict[str, Any]:
    response = client.post("/bugs/check", json={**report, **extra})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_file_anyway_proposes_canonical_possible_duplicate_and_confirm_files_it(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    github_api = FakeGitHubAPI()
    with verdict_client(ingested_store, fake_embedder, "duplicate", github_api=github_api) as c:
        plain = check(c, O2_REPORT)
        anyway = check(c, O2_REPORT, file_anyway=True)
        assert github_api.requests == []  # nothing filed without Confirm
        action = anyway["proposed_action"]
        confirmed = c.post(
            f"/actions/{action['action_id']}/confirm",
            json={"confirm_token": action["confirm_token"]},
        )
        replay = c.post(
            f"/actions/{action['action_id']}/confirm",
            json={"confirm_token": action["confirm_token"]},
        )

    assert plain["verdict"] == anyway["verdict"] == "duplicate"
    assert plain["proposed_action"] is None
    assert plain["file_anyway_available"] is True
    candidate_ids = {c["doc_id"] for c in anyway["candidates"]}
    preview = action["preview"]
    assert "possible-duplicate" in preview["labels"]
    [line] = [x for x in preview["body"].splitlines() if x.startswith("Possible duplicates:")]
    listed = [part.split(" ")[0] for part in line.removeprefix("Possible duplicates: ").split(", ")]
    assert anyway["duplicate_of"] in listed
    assert len(listed) == len(set(listed))
    # Chain members (labelled duplicate_of BUG-0002) are listed only by their canonical id.
    assert candidate_ids & {"BUG-0006", "BUG-0011"}, "expected a chain member as candidate"
    for member in ("BUG-0006", "BUG-0011"):
        if member in candidate_ids:
            assert member not in listed
            assert "BUG-0002" in listed

    assert confirmed.status_code == 200
    assert replay.status_code in (404, 409)
    [payload] = github_api.payloads()
    assert payload["title"] == preview["title"]
    assert payload["body"] == preview["body"]
    assert payload["labels"] == preview["labels"]


def test_file_anyway_with_filing_disabled(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    github_api = FakeGitHubAPI()
    with verdict_client(
        ingested_store, fake_embedder, "duplicate", github_api=github_api, actions_enabled=False
    ) as c:
        body = check(c, O2_REPORT, file_anyway=True)
    assert body["file_anyway_available"] is False
    action = body["proposed_action"]
    assert action["action_id"] is None and action["confirm_token"] is None
    assert action["actions_disabled_reason"]
    assert github_api.requests == []


def test_file_anyway_without_github_proposes_nothing(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    with verdict_client(ingested_store, fake_embedder, "duplicate") as c:
        body = check(c, O2_REPORT, file_anyway=True)
    assert body["proposed_action"] is None
    assert body["file_anyway_available"] is False


def test_file_anyway_with_daily_cap_reached(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    github_api = FakeGitHubAPI()
    with verdict_client(
        ingested_store, fake_embedder, "duplicate", github_api=github_api, actions_max_per_day=1
    ) as c:
        first = check(c, O2_REPORT, file_anyway=True)["proposed_action"]
        assert (
            c.post(
                f"/actions/{first['action_id']}/confirm",
                json={"confirm_token": first["confirm_token"]},
            ).status_code
            == 200
        )
        body = check(c, O2_REPORT, file_anyway=True)
    action = body["proposed_action"]
    assert action["confirm_token"] is None and action["action_id"] is None
    assert action["actions_disabled_reason"]
    assert len(github_api.requests) == 1


def _stable(body: dict[str, Any]) -> dict[str, Any]:
    """Response without the per-proposal random fields."""
    action = body.get("proposed_action")
    if action:
        action = {
            k: v for k, v in action.items() if k not in ("action_id", "confirm_token", "expires_at")
        }
    return {**body, "proposed_action": action}


@pytest.mark.parametrize("verdict", ["new", "possible_duplicate"])
def test_file_anyway_has_no_effect_on_other_verdicts(
    ingested_store: QdrantStore, fake_embedder: Embedder, verdict: str
) -> None:
    report = NEW_REPORT if verdict == "new" else O2_REPORT
    with verdict_client(ingested_store, fake_embedder, verdict, github_api=FakeGitHubAPI()) as c:
        plain = check(c, report)
        anyway = check(c, report, file_anyway=True)
    assert plain["verdict"] == verdict
    assert _stable(plain) == _stable(anyway)
    assert plain["file_anyway_available"] is False
    assert plain["proposed_action"]["confirm_token"]


@pytest.mark.parametrize("value", ["true", "yes", 1, 0, None, [], {}])
def test_file_anyway_must_be_strict_bool(
    ingested_store: QdrantStore, fake_embedder: Embedder, value: object
) -> None:
    with verdict_client(ingested_store, fake_embedder, "duplicate") as c:
        response = c.post("/bugs/check", json={**O2_REPORT, "file_anyway": value})
    detail = assert_input_free(response)
    assert ["body", "file_anyway"] in [e["loc"] for e in detail]


# =====================================================================================
# Pages
# =====================================================================================


def test_index_served_with_exact_headers(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.text == (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for name, value in PAGE_HEADERS.items():
        assert response.headers[name] == value


def test_voice_page_served_when_present(store: QdrantStore, fake_embedder: Embedder) -> None:
    with app_client(store, fake_embedder) as client:
        response = client.get("/voice")
    assert response.status_code == 200
    assert response.text == (STATIC_DIR / "voice.html").read_text(encoding="utf-8")


def test_voice_redirects_to_main_page_section_when_voice_html_absent(
    store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>")
    with app_client(store, fake_embedder, static_dir=tmp_path) as client:
        response = client.get("/voice")
        followed = client.get("/voice", follow_redirects=True)
    assert response.status_code == 307
    assert response.headers["location"] == "/#voice"
    for name, value in PAGE_HEADERS.items():
        assert response.headers[name] == value
    assert followed.status_code == 200


@pytest.mark.parametrize(
    ("files", "path", "status"),
    [
        ((), "/", 404),
        ((), "/voice", 404),
        (("voice.html",), "/", 404),
        (("voice.html",), "/voice", 200),
        (("index.html",), "/index.html", 404),
        (("index.html",), "/static/../index.html", 404),
    ],
)
def test_page_paths(
    store: QdrantStore,
    fake_embedder: Embedder,
    tmp_path: Path,
    files: tuple[str, ...],
    path: str,
    status: int,
) -> None:
    for name in files:
        (tmp_path / name).write_text("<!doctype html><title>x</title>")
    with app_client(store, fake_embedder, static_dir=tmp_path) as client:
        response = client.get(path)
    assert response.status_code == status


def test_trailing_slash_redirects_to_the_page_not_elsewhere(
    store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>")
    with app_client(store, fake_embedder, static_dir=tmp_path) as client:
        response = client.get("/voice/")
    # Starlette's redirect_slashes: same-origin path only, never an external host.
    assert response.status_code == 307
    assert httpx.URL(response.headers["location"]).path == "/voice"
    assert httpx.URL(response.headers["location"]).host in ("", "testserver")


def test_pages_404_without_static_dir(
    store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    with app_client(store, fake_embedder, static_dir=tmp_path / "missing") as client:
        assert client.get("/").status_code == 404
        assert client.get("/voice").status_code == 404
        assert client.get("/static/home.js").status_code == 404


# =====================================================================================
# /ask sources: platform and version
# =====================================================================================


def test_ask_sources_carry_exact_platform_and_version(
    ingested_store: QdrantStore, fake_embedder: Embedder, dataset: Any
) -> None:
    answer = LLMAnswer(answer="See [BUG-0001].", cited_ids=["BUG-0001"], insufficient_context=False)
    with app_client(ingested_store, fake_embedder, llm=FakeLLM({LLMAnswer: answer})) as client:
        body = client.post(
            "/ask", json={"question": "Why does my save get corrupted after cryo sleep on PS5?"}
        ).json()
    [source] = body["sources"]
    assert source["doc_id"] == "BUG-0001"
    assert source["platform"] == ["ps5"]
    assert source["version"] == "1.0.0"
    assert set(source) == {"doc_id", "doc_type", "title", "snippet", "score", "platform", "version"}


def test_ask_source_without_version_is_null(
    ingested_store: QdrantStore, fake_embedder: Embedder, dataset: Any
) -> None:
    docs = [d for d in dataset.docs] if hasattr(dataset, "docs") else []
    if not docs:
        pytest.skip("dataset has no markdown docs")
    doc_id = docs[0].id
    answer = LLMAnswer(answer=f"See [{doc_id}].", cited_ids=[doc_id], insufficient_context=False)
    with app_client(ingested_store, fake_embedder, llm=FakeLLM({LLMAnswer: answer})) as client:
        body = client.post("/ask", json={"question": docs[0].title}).json()
    sources = [s for s in body["sources"] if s["doc_id"] == doc_id]
    if not sources:
        pytest.skip("the doc was not retrieved by the fake embedder")
    assert sources[0]["version"] is None or isinstance(sources[0]["version"], str)
    assert isinstance(sources[0]["platform"], list)


# =====================================================================================
# Opt-in client-IP debug log (DEBUG_LOG_CLIENT_IP)
# =====================================================================================


def _ip_debug_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "client ip debug"]


def test_client_ip_debug_off_by_default(
    store: QdrantStore, fake_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with app_client(store, fake_embedder) as client:
        client.get("/health", headers={"X-Forwarded-For": "203.0.113.9"})
    assert _ip_debug_records(caplog) == []


def test_client_ip_debug_logs_only_peer_and_forwarded_for(
    store: QdrantStore, fake_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with app_client(store, fake_embedder, debug_log_client_ip=True) as client:
        client.post(
            f"/search?q={MARKER}",
            json={"query": MARKER},
            headers={
                "X-Forwarded-For": "203.0.113.9, 10.0.0.1",
                "Authorization": f"Bearer {MARKER}",
                "Cookie": f"s={MARKER}",
            },
        )
        client.get("/health")
    records = _ip_debug_records(caplog)
    assert len(records) == 2
    first, second = records
    assert first.peer == "testclient"  # type: ignore[attr-defined]
    assert first.x_forwarded_for == "203.0.113.9, 10.0.0.1"  # type: ignore[attr-defined]
    assert second.x_forwarded_for is None  # type: ignore[attr-defined]
    for record in records:
        assert MARKER not in repr(vars(record))
        assert "/search" not in repr(vars(record))


def test_client_ip_debug_truncates_forwarded_for(
    store: QdrantStore, fake_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    with app_client(store, fake_embedder, debug_log_client_ip=True) as client:
        client.get("/health", headers={"X-Forwarded-For": "1.2.3.4, " * 200})
    [record] = _ip_debug_records(caplog)
    assert len(record.x_forwarded_for) == 512  # type: ignore[attr-defined]
