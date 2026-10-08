"""GitHub issue client and Slack notifier on MockTransport: fixed hosts, no redirects, no
secret leakage, and Slack never affecting the issue result."""

import json
import logging
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from fakes import FAKE_REPO, FakeGitHubAPI
from studiodesk.actions.github import GitHubError, GitHubIssues
from studiodesk.actions.issue import ZERO_WIDTH_JOINER
from studiodesk.actions.slack import SKIPPED_MESSAGE, SlackNotifier, escape_slack_text
from studiodesk.config import Settings
from studiodesk.models.actions import CreatedIssue, IssueDraft, SlackStatus

TOKEN = "ghp_SECRETTOKENVALUE1234567890"  # noqa: S105 (fake)
WEBHOOK = "https://hooks.slack.com/services/T000/B000/SECRETWEBHOOKPART"
DRAFT = IssueDraft(title="Crash on load", body="Body", labels=["bug", "severity:high"])


def _all_log_text(caplog: pytest.LogCaptureFixture) -> str:
    return caplog.text + "".join(json.dumps(r.__dict__, default=str) for r in caplog.records)


def github(handler: Any, repo: str = FAKE_REPO) -> GitHubIssues:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GitHubIssues(http, repo, SecretStr(TOKEN), timeout_s=5)


# --------------------------------------------------------------------------- GitHub


def test_github_posts_only_to_fixed_issues_endpoint() -> None:
    api = FakeGitHubAPI()

    created = github(api).create(DRAFT)

    assert created == CreatedIssue(number=1, url=f"https://github.com/{FAKE_REPO}/issues/1")
    [request] = api.requests
    assert request.method == "POST"
    assert str(request.url) == f"https://api.github.com/repos/{FAKE_REPO}/issues"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert json.loads(request.content) == {
        "title": "Crash on load",
        "body": "Body",
        "labels": ["bug", "severity:high"],
    }


def test_github_does_not_follow_redirects() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(307, headers={"Location": "https://evil.example/steal"})

    # Even a client configured to follow redirects must not follow them for this call.
    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    client = GitHubIssues(http, FAKE_REPO, SecretStr(TOKEN), timeout_s=5)

    with pytest.raises(GitHubError, match="307"):
        client.create(DRAFT)
    assert len(seen) == 1
    assert seen[0].url.host == "api.github.com"


@pytest.mark.parametrize("status", [200, 202, 401, 403, 404, 422, 500])
def test_github_non_201_is_an_error(status: int) -> None:
    api = FakeGitHubAPI(status_code=status)
    with pytest.raises(GitHubError) as info:
        github(api).create(DRAFT)
    assert str(info.value) == f"github returned status {status}"


@pytest.mark.parametrize(
    "payload",
    [
        {"number": 1, "html_url": "https://github.com/other/repo/issues/1"},
        {"number": 1, "html_url": f"http://github.com/{FAKE_REPO}/issues/1"},
        {"number": 1, "html_url": f"https://github.com/{FAKE_REPO}/issues/2"},
        {"number": 1, "html_url": f"https://evil.example/{FAKE_REPO}/issues/1"},
        {"number": True, "html_url": f"https://github.com/{FAKE_REPO}/issues/True"},
        {"number": "1", "html_url": f"https://github.com/{FAKE_REPO}/issues/1"},
        {"number": 0, "html_url": f"https://github.com/{FAKE_REPO}/issues/0"},
        {"number": 1},
        [],
    ],
)
def test_github_unexpected_payload_is_an_error(payload: Any) -> None:
    client = github(lambda request: httpx.Response(201, json=payload))
    with pytest.raises(GitHubError, match="unexpected issue payload"):
        client.create(DRAFT)


def test_github_invalid_json_is_an_error() -> None:
    client = github(lambda request: httpx.Response(201, content=b"<html>"))
    with pytest.raises(GitHubError, match="invalid JSON"):
        client.create(DRAFT)


def test_github_transport_error_hides_details(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed with header Bearer {TOKEN}")

    with caplog.at_level(logging.DEBUG), pytest.raises(GitHubError) as info:
        github(handler).create(DRAFT)

    assert str(info.value) == "github request failed: ConnectError"
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert TOKEN not in _all_log_text(caplog)


def test_github_error_bodies_are_not_echoed() -> None:
    client = github(lambda request: httpx.Response(401, json={"message": f"bad {TOKEN}"}))
    with pytest.raises(GitHubError) as info:
        client.create(DRAFT)
    assert TOKEN not in str(info.value) and "bad" not in str(info.value)


def test_github_token_not_in_repr() -> None:
    client = github(FakeGitHubAPI())
    assert TOKEN not in repr(vars(client))


@pytest.mark.parametrize("repo", ["noslash", "a/b/c", "../x", "a/..", "/x", "a/", "./b"])
def test_github_rejects_bad_repo(repo: str) -> None:
    with pytest.raises(ValueError):
        github(FakeGitHubAPI(), repo=repo)


# --------------------------------------------------------------------------- Slack


class SlackRecorder:
    def __init__(self, response: Any = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response if response is not None else httpx.Response(200, text="ok")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def notifier(self, webhook: str | None = WEBHOOK) -> SlackNotifier:
        http = httpx.Client(transport=httpx.MockTransport(self))
        return SlackNotifier(http, SecretStr(webhook) if webhook else None, timeout_s=3)


ISSUE = CreatedIssue(number=7, url=f"https://github.com/{FAKE_REPO}/issues/7")


def test_slack_unset_is_skipped_without_http(caplog: pytest.LogCaptureFixture) -> None:
    rec = SlackRecorder()
    notifier = rec.notifier(webhook=None)

    with caplog.at_level(logging.INFO, logger="studiodesk.actions.slack"):
        status = notifier.notify_issue_created(ISSUE, DRAFT)

    assert status is SlackStatus.SKIPPED
    assert notifier.enabled is False
    assert rec.requests == []
    assert SKIPPED_MESSAGE in caplog.text


def test_slack_sends_plain_neutralised_text() -> None:
    rec = SlackRecorder()
    draft = IssueDraft(
        title="<!channel> crash @here <@U123> <https://evil.example|click> & *bold*",
        body="b",
        labels=["bug"],
    )

    status = rec.notifier().notify_issue_created(ISSUE, draft)

    assert status is SlackStatus.SENT
    [request] = rec.requests
    assert str(request.url) == WEBHOOK
    assert request.url.host == "hooks.slack.com"
    payload = json.loads(request.content)
    assert payload["mrkdwn"] is False
    assert set(payload) == {"text", "mrkdwn"}
    text = payload["text"]
    assert "<" not in text and ">" not in text
    assert "&lt;!channel&gt;" in text
    assert f"@{ZERO_WIDTH_JOINER}here" in text and "@here" not in text
    assert "&amp;" in text
    assert "#7" in text and ISSUE.url in text
    assert "Labels: bug" in text


def test_escape_slack_text() -> None:
    assert escape_slack_text("a<b>&@c") == f"a&lt;b&gt;&amp;@{ZERO_WIDTH_JOINER}c"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="no"),
        httpx.Response(302, headers={"Location": "https://evil.example"}),
        httpx.Response(404, text="no_service"),
        httpx.ConnectError("boom " + WEBHOOK),
        httpx.ReadTimeout("slow"),
    ],
)
def test_slack_failure_is_reported_never_raised(
    response: Any, caplog: pytest.LogCaptureFixture
) -> None:
    rec = SlackRecorder(response)

    with caplog.at_level(logging.DEBUG):
        status = rec.notifier().notify_issue_created(ISSUE, DRAFT)

    assert status is SlackStatus.FAILED
    assert len(rec.requests) == 1  # no redirect followed
    assert "SECRETWEBHOOKPART" not in _all_log_text(caplog)


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.slack.com/services/x",
        "https://hooks.slack.com.evil.example/services/x",
        "https://evil.example/services/x",
        "https://user:pw@hooks.slack.com/services/x",
        "https://hooks.slack.com:8443/services/x",
        "https://hooks.slack.com/",
        "https://hooks.slack.com",
        "ftp://hooks.slack.com/services/x",
    ],
)
def test_settings_reject_non_slack_webhooks(url: str) -> None:
    with pytest.raises(ValidationError) as info:
        Settings(_env_file=None, slack_webhook_url=url)
    assert "input_value" not in str(info.value)  # the URL is a secret: never echoed


def test_settings_accept_slack_webhook_and_empty() -> None:
    assert Settings(_env_file=None, slack_webhook_url=WEBHOOK).slack_webhook_url is not None
    assert Settings(_env_file=None, slack_webhook_url="  ").slack_webhook_url is None
