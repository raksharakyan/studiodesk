"""Minimal smoke tests for M6: the main page, /api/demo-info, "file anyway", the security
headers and the 422 handler (in-memory Qdrant, fake embedder/LLM/GitHub, no network).
The full suite is written by qa-agent."""

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from fakes import FAKE_REPO, FakeGitHubAPI, FakeLLM
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.models.bugs import CandidateJudgement, DuplicateJudgements
from studiodesk.vectorstore import QdrantStore
from studiodesk.voice import ElevenLabsClient

BUG_0002_PARAPHRASE = {
    "title": "Disconnected from co-op session when host opens the cargo hold",
    "description": "Whenever the host opens the cargo hold during co-op, every client "
    "gets disconnected with a desync error.",
    "platform": "pc",
    "version": "1.0.2",
}
QDRANT_URL = "https://example.cloud.qdrant.io"


def judge_yes(system: str, user: str) -> DuplicateJudgements:
    """Judge every candidate in the prompt a duplicate."""
    ids = [part.split('"')[0] for part in user.split('<document id="')[1:]]
    return DuplicateJudgements(
        judgements=[
            CandidateJudgement(candidate_id=i, is_duplicate=True, confidence=0.9, reason="r")
            for i in ids
        ]
    )


def run(
    store: QdrantStore,
    embedder: Embedder,
    *,
    static_dir: Path | None = None,
    llm: FakeLLM | None = None,
    github_api: FakeGitHubAPI | None = None,
    voice: ElevenLabsClient | None = None,
    **overrides: object,
) -> Iterator[TestClient]:
    fields: dict[str, object] = {"app_env": "test", **overrides}
    settings = Settings(_env_file=None, **fields)  # type: ignore[arg-type]
    extra: dict[str, object] = {"static_dir": static_dir} if static_dir is not None else {}
    app = create_app(
        settings,
        embedder=embedder,
        store=store,
        llm=llm,
        github=github_api.client() if github_api else None,
        voice=voice,
        **extra,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        yield client


def test_index_page_served_with_csp(
    store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>StudioDesk</title>")
    for client in run(store, fake_embedder, static_dir=tmp_path):
        page = client.get("/")
        voice = client.get("/voice", follow_redirects=False)

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in page.headers["Content-Security-Policy"]
    assert "microphone=(self)" in page.headers["Permissions-Policy"]
    assert voice.status_code == 307  # no voice.html: point to the main page's section
    assert voice.headers["location"] == "/#voice"


def test_index_missing_is_404(store: QdrantStore, fake_embedder: Embedder, tmp_path: Path) -> None:
    for client in run(store, fake_embedder, static_dir=tmp_path):
        assert client.get("/").status_code == 404
        assert client.get("/voice").status_code == 404


def test_demo_info_shape_and_no_secrets(store: QdrantStore, fake_embedder: Embedder) -> None:
    voice = ElevenLabsClient(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
        SecretStr("xi-secret-not-real"),
        timeout_s=5,
    )
    for client in run(
        store,
        fake_embedder,
        github_api=FakeGitHubAPI(),
        voice=voice,
        github_repo=FAKE_REPO,
        elevenlabs_agent_id="agent_secret123",
        actions_max_per_day=5,
        voice_max_sessions_per_day=3,
    ):
        response = client.get("/api/demo-info")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json() == {
        "game": "Starfall Outpost",
        "source_repo_url": "https://github.com/raksharakyan/studiodesk",
        "sandbox_repo": FAKE_REPO,
        "sandbox_issues_url": f"https://github.com/{FAKE_REPO}/issues",
        "actions": {"enabled": True, "configured": True, "max_per_day": 5, "remaining_today": 5},
        "voice": {"enabled": True, "configured": True, "max_per_day": 3, "remaining_today": 3},
    }
    for secret in ("xi-secret", "test-token-not-real", "agent_secret123"):
        assert secret not in response.text


def test_demo_info_prod_defaults_off(store: QdrantStore, fake_embedder: Embedder) -> None:
    for client in run(store, fake_embedder, app_env="prod", qdrant_url=QDRANT_URL):
        body = client.get("/api/demo-info").json()
    assert body["sandbox_repo"] is None and body["sandbox_issues_url"] is None
    assert body["actions"]["enabled"] is False and body["actions"]["configured"] is False
    assert body["voice"]["enabled"] is False and body["voice"]["configured"] is False


@pytest.mark.parametrize("file_anyway", [False, True])
def test_file_anyway_on_duplicate(
    ingested_store: QdrantStore, fake_embedder: Embedder, file_anyway: bool
) -> None:
    github_api = FakeGitHubAPI()
    payload = {**BUG_0002_PARAPHRASE, **({"file_anyway": True} if file_anyway else {})}
    for client in run(
        ingested_store,
        fake_embedder,
        llm=FakeLLM({DuplicateJudgements: judge_yes}),  # type: ignore[dict-item]
        github_api=github_api,
        dup_candidate_threshold=0.0,
        dup_auto_threshold=0.0,
    ):
        body = client.post("/bugs/check", json=payload).json()

    assert body["verdict"] == "duplicate"
    assert body["file_anyway_available"] is True
    assert github_api.requests == []
    if not file_anyway:
        assert body["proposed_action"] is None
        return
    action = body["proposed_action"]
    assert action["confirm_token"]
    preview = action["preview"]
    assert "possible-duplicate" in preview["labels"]
    assert f"Possible duplicates: {body['duplicate_of']} (score " in preview["body"]


def test_file_anyway_must_be_a_real_boolean(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    for client in run(ingested_store, fake_embedder, llm=FakeLLM({})):
        response = client.post("/bugs/check", json={**BUG_0002_PARAPHRASE, "file_anyway": "yes"})
    assert response.status_code == 422


def test_baseline_headers_everywhere_and_prod_extras(
    store: QdrantStore, fake_embedder: Embedder
) -> None:
    for client in run(store, fake_embedder):
        dev = client.get("/health")
        missing = client.get("/nope")
    for client in run(store, fake_embedder, app_env="prod", qdrant_url=QDRANT_URL):
        prod = client.get("/health")

    for response in (dev, missing, prod):
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "Strict-Transport-Security" not in dev.headers
    assert "Cross-Origin-Opener-Policy" not in dev.headers
    assert prod.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert prod.headers["Cross-Origin-Opener-Policy"] == "same-origin"


def test_422_never_echoes_input(store: QdrantStore, fake_embedder: Embedder) -> None:
    marker = "SECRET-INPUT-MARKER"
    for client in run(store, fake_embedder):
        response = client.post("/search", json={"query": marker * 1000, marker: 1})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors
    for error in errors:
        assert set(error) == {"loc", "type", "msg"}
        assert error["msg"] == "Invalid value"
    assert "SECRET-INPUT-MARKER" * 2 not in response.text
    assert "max_length" not in response.text  # no ctx values


def test_ask_sources_carry_platform_and_version(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    from studiodesk.models.answer import LLMAnswer

    answer = LLMAnswer(answer="See [BUG-0001].", cited_ids=["BUG-0001"], insufficient_context=False)
    for client in run(ingested_store, fake_embedder, llm=FakeLLM({LLMAnswer: answer})):
        body = client.post(
            "/ask", json={"question": "Why does my save get corrupted after cryo sleep on PS5?"}
        )
    [source] = body.json()["sources"]
    assert source["doc_id"] == "BUG-0001"
    assert isinstance(source["platform"], list) and all(
        isinstance(p, str) for p in source["platform"]
    )
    assert source["version"] is None or isinstance(source["version"], str)
