"""Opt-in live smoke tests against the real configured LLM provider.

Skipped unless STUDIODESK_LIVE=1 (see conftest). They never read `.env`: keys must already
be exported in the environment (Settings(_env_file=None) reads only process env vars).
Retrieval uses the in-memory store with the fake embedder, and GitHub/Slack are never
contacted (no GitHub client is injected and no webhook is set), so only the LLM is live.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.vectorstore import QdrantStore

pytestmark = pytest.mark.live


@pytest.fixture
def live_client(
    ingested_store: QdrantStore, fake_embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    for name in ("GITHUB_TOKEN", "GITHUB_REPO", "SLACK_WEBHOOK_URL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, app_env="test")
    if settings.llm_api_key is None:
        pytest.skip("no API key for the selected LLM provider in the environment")
    with TestClient(create_app(settings, embedder=fake_embedder, store=ingested_store)) as c:
        yield c


def test_live_ask_cites_retrieved_documents(live_client: TestClient) -> None:
    response = live_client.post(
        "/ask",
        json={"question": "Why does my save get corrupted after cryo sleep on PS5, is it fixed?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"]
    assert all(
        s["doc_id"].split("-")[0] in {"BUG", "CRASH", "PATCH", "DOC"} for s in body["sources"]
    )


def test_live_bugs_check_known_duplicate(live_client: TestClient) -> None:
    response = live_client.post(
        "/bugs/check",
        json={
            "title": "Disconnected from co-op session when host opens the cargo hold",
            "description": "Whenever the host opens the cargo hold during co-op, every client "
            "gets disconnected with a desync error.",
            "platform": "pc",
            "version": "1.0.2",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verdict"] in {"duplicate", "possible_duplicate", "new"}
    assert body["proposed_action"] is None  # GitHub is not configured here
