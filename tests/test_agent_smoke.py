"""Minimal smoke tests for /ask, /bugs/check and confirm (FakeLLM, fake GitHub, in-memory
Qdrant with the fake embedder). The full suite is written by qa-agent."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from fakes import FAKE_REPO, FakeGitHubAPI, FakeLLM
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.models.answer import LLMAnswer
from studiodesk.models.bugs import CandidateJudgement, DuplicateJudgements
from studiodesk.vectorstore import QdrantStore

BUG_0002_PARAPHRASE = {
    "title": "Disconnected from co-op session when host opens the cargo hold",
    "description": "Whenever the host opens the cargo hold during co-op, every client "
    "gets disconnected with a desync error.",
    "platform": "pc",
    "version": "1.0.2",
}
NOVEL_BUG = {
    "title": "Photo mode camera ignores inverted Y axis",
    "description": "In photo mode the free camera ignores the inverted Y axis setting.",
    "steps_to_reproduce": ["Enable inverted Y", "Open photo mode", "Move the camera up"],
    "platform": "ps5",
    "version": "1.0.2",
}


def judge(is_duplicate: bool) -> object:
    """Responder that judges every candidate in the prompt the same way."""

    def respond(system: str, user: str) -> DuplicateJudgements:
        ids = [part.split('"')[0] for part in user.split('<document id="')[1:]]
        return DuplicateJudgements(
            judgements=[
                CandidateJudgement(
                    candidate_id=doc_id, is_duplicate=is_duplicate, confidence=0.9, reason="r"
                )
                for doc_id in ids
            ]
        )

    return respond


@pytest.fixture
def github_api() -> FakeGitHubAPI:
    return FakeGitHubAPI()


def make_client(
    ingested_store: QdrantStore,
    fake_embedder: Embedder,
    llm: FakeLLM,
    github_api: FakeGitHubAPI,
    **overrides: object,
) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, app_env="test", **overrides)  # type: ignore[arg-type]
    app = create_app(
        settings,
        embedder=fake_embedder,
        store=ingested_store,
        llm=llm,
        github=github_api.client(),
    )
    with TestClient(app) as client:
        yield client


def test_ask_drops_citations_that_were_not_retrieved(
    ingested_store: QdrantStore, fake_embedder: Embedder, github_api: FakeGitHubAPI
) -> None:
    llm = FakeLLM(
        {
            LLMAnswer: LLMAnswer(
                answer="Fixed in 1.0.1 [BUG-0001], see also [BUG-9999].",
                cited_ids=["BUG-0001", "BUG-9999"],
                insufficient_context=False,
            )
        }
    )
    for client in make_client(ingested_store, fake_embedder, llm, github_api):
        response = client.post(
            "/ask",
            json={"question": "Why does my save get corrupted after cryo sleep on PS5?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Fixed in 1.0.1 [BUG-0001], see also [unverified source]."
    assert body["removed_citations"] == 1
    assert [source["doc_id"] for source in body["sources"]] == ["BUG-0001"]
    assert "<user_question>" in llm.calls[0].user_content


def test_bugs_check_duplicate_proposes_nothing(
    ingested_store: QdrantStore, fake_embedder: Embedder, github_api: FakeGitHubAPI
) -> None:
    llm = FakeLLM({DuplicateJudgements: judge(True)})  # type: ignore[dict-item]
    clients = make_client(
        ingested_store,
        fake_embedder,
        llm,
        github_api,
        dup_candidate_threshold=0.0,
        dup_auto_threshold=0.0,
    )
    for client in clients:
        response = client.post("/bugs/check", json=BUG_0002_PARAPHRASE)

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "duplicate"
    assert body["duplicate_of"] is not None
    assert body["proposed_action"] is None
    assert github_api.requests == []


def test_new_bug_confirm_files_issue_once(
    ingested_store: QdrantStore, fake_embedder: Embedder, github_api: FakeGitHubAPI
) -> None:
    llm = FakeLLM({DuplicateJudgements: judge(False)})  # type: ignore[dict-item]
    for client in make_client(ingested_store, fake_embedder, llm, github_api):
        checked = client.post(
            "/bugs/check", json={**NOVEL_BUG, "title": "@team " + NOVEL_BUG["title"]}
        )
        assert checked.status_code == 200
        action = checked.json()["proposed_action"]
        assert checked.json()["verdict"] == "new"
        assert github_api.requests == []

        url = f"/actions/{action['action_id']}/confirm"
        confirmed = client.post(url, json={"confirm_token": action["confirm_token"]})
        again = client.post(url, json={"confirm_token": action["confirm_token"]})

    assert confirmed.status_code == 200
    assert confirmed.json() == {
        "status": "confirmed",
        "issue_number": 1,
        "issue_url": f"https://github.com/{FAKE_REPO}/issues/1",
        "slack": "skipped",
    }
    assert again.status_code == 409
    assert len(github_api.requests) == 1
    assert str(github_api.requests[0].url) == f"https://api.github.com/repos/{FAKE_REPO}/issues"
    payload = github_api.payloads()[0]
    assert payload == action["preview"]  # confirm sends exactly the stored draft
    assert "@team" not in str(payload["title"])
    assert "bug" in action["preview"]["labels"]
