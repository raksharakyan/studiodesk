"""Contract between the real API and the main page's helpers.

Real `GET /api/demo-info` and `POST /bugs/check` responses (TestClient, fakes, no network)
are written to a JSON file and fed to tests/js/home_contract.mjs, which runs home.js's
`limitText`, `sandboxLink`, `projectLinks` and voice.js's `canFileAnyway`,
`isProposalReady`, `toAgentBugResult` and `buildFileAnywayBody` on them. Then the
"file it anyway" body the page would send is posted back to the real API, and the issue
is filed only after Confirm. Skipped when node is not installed.
"""

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME_JS = REPO_ROOT / "src" / "studiodesk" / "static" / "home.js"
CONTRACT = REPO_ROOT / "tests" / "js" / "home_contract.mjs"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

QDRANT_URL = "https://example.cloud.qdrant.io"
O2_REPORT = {
    "title": "Oxygen drains for all clients after the host leaves co-op",
    "description": "When the host quits and host migration happens, everyone except the new "
    "host suffocates although the reactor shows 100 percent. We all died of no air.",
    "platform": "pc",
    "version": "1.0.2",
}


def judge_yes(system: str, user: str) -> DuplicateJudgements:
    ids = [part.split('"')[0] for part in user.split('<document id="')[1:]]
    return DuplicateJudgements(
        judgements=[
            CandidateJudgement(candidate_id=i, is_duplicate=True, confidence=0.9, reason="r")
            for i in ids
        ]
    )


def make_app(
    store: QdrantStore,
    embedder: Embedder,
    *,
    github_api: FakeGitHubAPI | None = None,
    voice: ElevenLabsClient | None = None,
    **fields: object,
) -> Any:
    settings = Settings(_env_file=None, **{"app_env": "test", **fields})  # type: ignore[arg-type]
    return create_app(
        settings,
        embedder=embedder,
        store=store,
        llm=FakeLLM({DuplicateJudgements: judge_yes}),  # type: ignore[dict-item]
        github=github_api.client() if github_api else None,
        voice=voice,
    )


def run_contract(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    assert NODE is not None
    data = tmp_path / "contract.json"
    data.write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [NODE, str(CONTRACT), str(HOME_JS), str(data)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result: dict[str, Any] = json.loads(proc.stdout)
    return result


def ok_voice() -> ElevenLabsClient:
    handler: Callable[[httpx.Request], httpx.Response] = lambda r: httpx.Response(  # noqa: E731
        200, json={"signed_url": "wss://api.elevenlabs.io/v1/convai/conversation?x=1"}
    )
    return ElevenLabsClient(
        httpx.Client(transport=httpx.MockTransport(handler)), SecretStr("xi-not-real"), timeout_s=5
    )


def test_demo_info_contract(
    ingested_store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    infos: dict[str, Any] = {}

    configured = make_app(
        ingested_store,
        fake_embedder,
        github_api=FakeGitHubAPI(),
        voice=ok_voice(),
        github_repo=FAKE_REPO,
        elevenlabs_agent_id="agent_contract",
        actions_max_per_day=3,
        voice_max_sessions_per_day=1,
        dup_candidate_threshold=1.0,
        dup_auto_threshold=1.0,
    )
    with TestClient(configured) as client:
        infos["fresh"] = client.get("/api/demo-info").json()
        action = client.post("/bugs/check", json=O2_REPORT).json()["proposed_action"]
        client.post(
            f"/actions/{action['action_id']}/confirm",
            json={"confirm_token": action["confirm_token"]},
        )
        assert client.get("/voice/session").status_code == 200
        infos["used"] = client.get("/api/demo-info").json()

    with TestClient(
        make_app(ingested_store, fake_embedder, app_env="prod", qdrant_url=QDRANT_URL)
    ) as client:
        infos["prod_default"] = client.get("/api/demo-info").json()

    with TestClient(make_app(ingested_store, fake_embedder)) as client:
        infos["unconfigured"] = client.get("/api/demo-info").json()

    out = run_contract(tmp_path, {"demo_infos": infos})["demo_infos"]

    assert out["fresh"]["actions"] == "Issue filing: 3 of 3 left today."
    assert out["fresh"]["voice"] == "Voice calls: 1 of 1 left today."
    assert out["fresh"]["sandbox"] == {
        "href": f"https://github.com/{FAKE_REPO}/issues",
        "text": FAKE_REPO,
    }
    assert out["fresh"]["links"] == {
        "repo": "https://github.com/raksharakyan/studiodesk",
        "evals": "https://github.com/raksharakyan/studiodesk/blob/main/evals/reports/latest.md",
        "security": "https://github.com/raksharakyan/studiodesk/tree/main/docs/security",
    }
    assert out["used"]["actions"] == "Issue filing: 2 of 3 left today."
    assert out["used"]["voice"] == (
        "Voice calls: today's limit of 1 is used up. Please try again tomorrow."
    )
    assert out["prod_default"]["actions"] == "Issue filing: turned off on this server right now."
    assert out["prod_default"]["voice"] == "Voice calls: turned off on this server right now."
    assert out["prod_default"]["sandbox"] is None
    assert out["unconfigured"]["actions"] == "Issue filing: not set up on this server."
    assert out["unconfigured"]["voice"] == "Voice calls: not set up on this server."
    assert out["unconfigured"]["sandbox"] is None


def test_file_anyway_contract_still_needs_confirm(
    ingested_store: QdrantStore, fake_embedder: Embedder, tmp_path: Path
) -> None:
    github_api = FakeGitHubAPI()
    app = make_app(
        ingested_store,
        fake_embedder,
        github_api=github_api,
        github_repo=FAKE_REPO,
        dup_candidate_threshold=0.0,
        dup_auto_threshold=0.0,
    )
    disabled_app = make_app(
        ingested_store,
        fake_embedder,
        github_api=FakeGitHubAPI(),
        github_repo=FAKE_REPO,
        actions_enabled=False,
        dup_candidate_threshold=0.0,
        dup_auto_threshold=0.0,
    )
    with TestClient(disabled_app) as client:
        disabled = client.post("/bugs/check", json=O2_REPORT).json()

    with TestClient(app) as client:
        duplicate = client.post("/bugs/check", json=O2_REPORT).json()
        # The page re-sends exactly what was judged, plus file_anyway: true.
        page = run_contract(
            tmp_path,
            {
                "bug_responses": {"duplicate": duplicate, "disabled": disabled},
                "last_check_body": O2_REPORT,
            },
        )
        anyway_body = page["file_anyway_body"]
        assert anyway_body == {**O2_REPORT, "file_anyway": True}
        response = client.post("/bugs/check", json=anyway_body)
        assert response.status_code == 200
        anyway = response.json()
        assert github_api.requests == []  # proposing never files

        after = run_contract(tmp_path, {"bug_responses": {"anyway": anyway}})["bug_responses"]
        action = anyway["proposed_action"]
        confirmed = client.post(
            f"/actions/{action['action_id']}/confirm",
            json={"confirm_token": action["confirm_token"]},
        )

    seen = page["bug_responses"]
    assert duplicate["verdict"] == "duplicate"
    assert seen["duplicate"]["can_file_anyway"] is True
    assert seen["duplicate"]["proposal_ready"] is False
    assert seen["disabled"]["can_file_anyway"] is False
    assert after["anyway"]["can_file_anyway"] is False
    assert after["anyway"]["proposal_ready"] is True
    agent = after["anyway"]["agent_result"]
    assert agent["proposal_ready"] is True
    assert "confirm_token" not in json.dumps(agent) and action["confirm_token"] not in json.dumps(
        agent
    )
    assert confirmed.status_code == 200
    assert len(github_api.requests) == 1
    [payload] = github_api.payloads()
    assert "possible-duplicate" in payload["labels"]  # type: ignore[operator]
