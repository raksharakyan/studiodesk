"""scripts/setup_issue_labels.py against a fake GitHub labels API (MockTransport).

`main()` reads Settings from the environment and `.env` in the working directory, so every
test runs in an empty temporary directory with the GitHub variables set explicitly.
"""

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from studiodesk.actions.labels import LABEL_ALLOWLIST, LABEL_SPECS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "setup_issue_labels.py"
REPO = "studio/sandbox"
TOKEN = "ghp_LABELSCRIPTFAKETOKEN0000000000000"  # noqa: S105 (fake)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("setup_issue_labels", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeLabelsAPI:
    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    list_status: int = 200
    create_status: int | None = None
    create_text: str = ""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.host == "api.github.com"
        assert request.url.path == f"/repos/{REPO}/labels"
        if request.method == "GET":
            if self.list_status != 200:
                return httpx.Response(self.list_status, json={"message": TOKEN})
            per_page = int(request.url.params["per_page"])
            page = int(request.url.params["page"])
            names = sorted(self.labels)
            batch = names[(page - 1) * per_page : page * per_page]
            return httpx.Response(200, json=[self.labels[n] for n in batch])
        if request.method == "POST":
            if self.create_status is not None:
                return httpx.Response(self.create_status, text=self.create_text)
            data = json.loads(request.content)
            self.labels[data["name"]] = data
            return httpx.Response(201, json=data)
        return httpx.Response(405)

    def methods(self) -> list[str]:
        return [r.method for r in self.requests]

    def http(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def test_sync_creates_only_missing_and_is_idempotent(script: ModuleType) -> None:
    custom = {"name": "bug", "color": "000000", "description": "owner's own"}
    api = FakeLabelsAPI(labels={"bug": custom, "wontfix": {"name": "wontfix"}})

    created, kept = script.sync_labels(api.http(), REPO, SecretStr(TOKEN))

    assert kept == ["bug"]
    assert set(created) == LABEL_ALLOWLIST - {"bug"}
    assert api.labels["bug"] == custom  # never modified
    assert "wontfix" in api.labels  # never deleted
    assert set(api.methods()) == {"GET", "POST"}
    posted = [json.loads(r.content) for r in api.requests if r.method == "POST"]
    specs = {s.name: s for s in LABEL_SPECS}
    for item in posted:
        spec = specs[item["name"]]
        assert item == {"name": spec.name, "color": spec.color, "description": spec.description}
    assert all(r.headers["Authorization"] == f"Bearer {TOKEN}" for r in api.requests)

    api.requests.clear()
    created_again, kept_again = script.sync_labels(api.http(), REPO, SecretStr(TOKEN))

    assert created_again == []
    assert set(kept_again) == LABEL_ALLOWLIST
    assert api.methods() == ["GET"]


def test_sync_paginates(script: ModuleType) -> None:
    labels = {f"aaa-{i:03d}": {"name": f"aaa-{i:03d}"} for i in range(150)}
    labels["severity:low"] = {"name": "severity:low"}  # sorts onto page 2
    api = FakeLabelsAPI(labels=labels)

    created, kept = script.sync_labels(api.http(), REPO, SecretStr(TOKEN))

    assert kept == ["severity:low"]
    assert "severity:low" not in created
    gets = [r for r in api.requests if r.method == "GET"]
    assert [r.url.params["page"] for r in gets] == ["1", "2"]


def test_already_exists_race_counts_as_success(script: ModuleType) -> None:
    api = FakeLabelsAPI(create_status=422, create_text='{"errors":[{"code":"already_exists"}]}')
    created, _ = script.sync_labels(api.http(), REPO, SecretStr(TOKEN))
    assert set(created) == LABEL_ALLOWLIST


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"list_status": 401}, "listing labels returned 401"),
        ({"create_status": 403}, "returned 403"),
        ({"create_status": 422, "create_text": "invalid color"}, "returned 422"),
    ],
)
def test_sync_errors_hide_details(script: ModuleType, kwargs: dict[str, Any], match: str) -> None:
    api = FakeLabelsAPI(**kwargs)
    with pytest.raises(script.LabelSyncError, match=match) as info:
        script.sync_labels(api.http(), REPO, SecretStr(TOKEN))
    assert TOKEN not in str(info.value)


def test_transport_error_hides_details(script: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect with {TOKEN}")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(script.LabelSyncError) as info:
        script.sync_labels(http, REPO, SecretStr(TOKEN))
    assert str(info.value) == "listing labels failed: ConnectError"


# --------------------------------------------------------------------------- main()


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[pytest.MonkeyPatch]:
    """Empty cwd (no .env) and no GitHub variables from the shell."""
    monkeypatch.chdir(tmp_path)
    for name in ("GITHUB_TOKEN", "GITHUB_REPO", "SLACK_WEBHOOK_URL", "APP_ENV"):
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch


def _route_http(monkeypatch: pytest.MonkeyPatch, script: ModuleType, api: FakeLabelsAPI) -> None:
    original = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        assert kwargs.get("follow_redirects") is False
        assert kwargs.get("trust_env") is False  # no proxy/netrc from the environment
        return original(transport=httpx.MockTransport(api), **kwargs)

    monkeypatch.setattr(script.httpx, "Client", client)


def test_main_without_token_exits_nonzero(
    script: ModuleType, isolated: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    isolated.setenv("GITHUB_REPO", REPO)
    api = FakeLabelsAPI()
    _route_http(isolated, script, api)

    assert script.main() == 2

    out = capsys.readouterr()
    assert "GITHUB_TOKEN" in out.err
    assert api.requests == []


def test_main_without_repo_exits_nonzero(
    script: ModuleType, isolated: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    isolated.setenv("GITHUB_TOKEN", TOKEN)
    api = FakeLabelsAPI()
    _route_http(isolated, script, api)

    assert script.main() == 2
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err
    assert api.requests == []


def test_main_invalid_settings_do_not_echo_values(
    script: ModuleType, isolated: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    isolated.setenv("GITHUB_TOKEN", TOKEN)
    isolated.setenv("GITHUB_REPO", f"not a repo {TOKEN}")

    assert script.main() == 2
    out = capsys.readouterr()
    assert "github_repo" in out.err
    assert TOKEN not in out.out + out.err


def test_main_success_and_failure_never_print_token(
    script: ModuleType, isolated: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    isolated.setenv("GITHUB_TOKEN", TOKEN)
    isolated.setenv("GITHUB_REPO", REPO)
    api = FakeLabelsAPI(labels={"bug": {"name": "bug"}})
    _route_http(isolated, script, api)

    assert script.main() == 0
    first = capsys.readouterr()
    assert f"Repository: {REPO}" in first.out
    assert f"Created ({len(LABEL_ALLOWLIST) - 1})" in first.out
    assert "Already present (1): bug" in first.out

    assert script.main() == 0
    second = capsys.readouterr()
    assert "Created (0): -" in second.out

    api.list_status = 500
    assert script.main() == 1
    third = capsys.readouterr()
    assert "error: listing labels returned 500" in third.err

    for out in (first, second, third):
        assert TOKEN not in out.out + out.err


def test_script_subprocess_without_token_exits_2(tmp_path: Path) -> None:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    result = subprocess.run(  # noqa: S603 - fixed interpreter and script path
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "set GITHUB_TOKEN and GITHUB_REPO" in result.stderr
    assert result.stdout == ""
