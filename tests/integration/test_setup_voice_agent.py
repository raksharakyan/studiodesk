"""scripts/setup_voice_agent.py against a fake ElevenLabs Agents API (MockTransport).

`run()` takes Settings and an injected httpx client; `main()` reads Settings from the
process environment and `.env` in the working directory, so those tests run in an empty
temporary directory with the ElevenLabs variables set or removed explicitly.
"""

import importlib.util
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from studiodesk.config import Settings
from studiodesk.voice.agent_config import load_agent_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "setup_voice_agent.py"
AGENT_JSON = REPO_ROOT / "voice" / "agent.json"
API_KEY = "xi-setup-key-not-real-0123456789abcdef"
TOOL_NAMES = [tool.name for tool in load_agent_config(AGENT_JSON).tools]
VOICE_ENV = (
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_AGENT_ID",
    "ELEVENLABS_TIMEOUT_S",
    "VOICE_LLM",
    "VOICE_RATE_LIMIT",
)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("setup_voice_agent", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeElevenLabs:
    """In-memory ElevenLabs workspace: tools and agents, with a paginated tool list."""

    tools: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> tool_config
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> payload
    page_size: int = 100
    fail: tuple[str, str, int, Any] | None = None  # (method, path prefix, status, body)
    requests: list[httpx.Request] = field(default_factory=list)
    _next: int = 0

    def _new_id(self, prefix: str) -> str:
        self._next += 1
        return f"{prefix}_{self._next:04d}"

    def add_tool(self, name: str) -> str:
        tool_id = self._new_id("tool")
        self.tools[tool_id] = {"name": name, "type": "client"}
        return tool_id

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.scheme == "https"
        assert request.url.host == "api.elevenlabs.io"
        assert request.url.port is None
        assert request.headers["xi-api-key"] == API_KEY
        method, path = request.method, request.url.path
        if self.fail and method == self.fail[0] and path.startswith(self.fail[1]):
            return httpx.Response(self.fail[2], json=self.fail[3])
        body = json.loads(request.content) if request.content else None

        if method == "GET" and path == "/v1/convai/tools":
            ids = sorted(self.tools)
            start = int(request.url.params.get("cursor", "0"))
            page = ids[start : start + self.page_size]
            more = start + self.page_size < len(ids)
            return httpx.Response(
                200,
                json={
                    "tools": [{"id": i, "tool_config": self.tools[i]} for i in page],
                    "has_more": more,
                    "next_cursor": str(start + self.page_size) if more else None,
                },
            )
        if method == "POST" and path == "/v1/convai/tools":
            tool_id = self._new_id("tool")
            self.tools[tool_id] = body["tool_config"]
            return httpx.Response(200, json={"id": tool_id})
        if method == "PATCH" and path.startswith("/v1/convai/tools/"):
            tool_id = path.rsplit("/", 1)[1]
            if tool_id not in self.tools:
                return httpx.Response(404, json={"detail": "tool not found"})
            self.tools[tool_id] = body["tool_config"]
            return httpx.Response(200, json={"id": tool_id})
        if method == "POST" and path == "/v1/convai/agents/create":
            agent_id = self._new_id("agent")
            self.agents[agent_id] = body
            return httpx.Response(200, json={"agent_id": agent_id})
        if method == "PATCH" and path.startswith("/v1/convai/agents/"):
            agent_id = path.rsplit("/", 1)[1]
            if agent_id not in self.agents:
                return httpx.Response(404, json={"detail": "agent not found"})
            self.agents[agent_id] = body
            return httpx.Response(200, json={"agent_id": agent_id})
        raise AssertionError(f"unexpected request {method} {path}")

    def http(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def calls(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]

    def tool_names(self) -> list[str]:
        return sorted(cfg["name"] for cfg in self.tools.values())


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"elevenlabs_api_key": API_KEY}
    values.update(overrides)
    return Settings(_env_file=None, app_env="test", **values)


def run(
    script: ModuleType, fake: FakeElevenLabs, argv: list[str], cfg: Settings | None = None
) -> int:
    args = script.parse_args(argv)
    code: int = script.run(args, cfg or settings(), http=fake.http())
    return code


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch, script: ModuleType) -> list[str]:
    """Fail any real HTTP send or client build; returns the list of attempts."""
    attempts: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append("network")
        raise AssertionError("network call attempted")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(script, "build_http_client", refuse)
    return attempts


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Empty cwd (no `.env`) and no ElevenLabs variables in the environment."""
    for name in VOICE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# --- dry run -------------------------------------------------------------------------


def test_dry_run_without_key_makes_no_calls(
    script: ModuleType,
    clean_env: Path,
    no_network: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert script.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert no_network == []
    for name in TOOL_NAMES:
        assert f"# tool {name}:" in out
    assert "# agent: POST /v1/convai/agents/create" in out
    assert '"enable_auth": true' in out


def test_dry_run_with_key_and_agent_never_prints_key(
    script: ModuleType,
    clean_env: Path,
    no_network: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", API_KEY)
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent_existing")
    assert script.main(["--dry-run", "--allow-host", "localhost"]) == 0
    captured = capsys.readouterr()
    assert no_network == []
    assert API_KEY not in captured.out + captured.err
    assert "# agent: PATCH /v1/convai/agents/agent_existing" in captured.out
    assert '"hostname": "localhost"' in captured.out


def test_main_without_key_exits_2_without_network(
    script: ModuleType,
    clean_env: Path,
    no_network: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert script.main([]) == 2
    assert "ELEVENLABS_API_KEY" in capsys.readouterr().err
    assert no_network == []


def test_main_reports_invalid_settings_without_values(
    script: ModuleType,
    clean_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "bad id/with-secret-ish-value")
    assert script.main(["--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "elevenlabs_agent_id" in err
    assert "secret-ish-value" not in err


# --- first and second run ------------------------------------------------------------


def test_first_run_creates_tools_and_agent(
    script: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeElevenLabs()
    assert run(script, fake, []) == 0

    assert fake.calls() == [
        ("GET", "/v1/convai/tools"),
        *[("POST", "/v1/convai/tools")] * len(TOOL_NAMES),
        ("POST", "/v1/convai/agents/create"),
    ]
    assert fake.tool_names() == sorted(TOOL_NAMES)
    [(agent_id, agent)] = fake.agents.items()
    assert agent["platform_settings"]["auth"]["enable_auth"] is True
    prompt = agent["conversation_config"]["agent"]["prompt"]
    assert sorted(prompt["tool_ids"]) == sorted(fake.tools)
    assert prompt["llm"] == Settings(_env_file=None).voice_llm
    out = capsys.readouterr().out
    assert f"ELEVENLABS_AGENT_ID={agent_id}" in out
    assert API_KEY not in out


def test_second_run_patches_without_duplicates(
    script: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeElevenLabs()
    assert run(script, fake, []) == 0
    [agent_id] = fake.agents
    tools_before = dict(fake.tools)
    fake.requests.clear()

    assert run(script, fake, [], settings(elevenlabs_agent_id=agent_id)) == 0

    assert fake.calls() == [
        ("GET", "/v1/convai/tools"),
        *[("PATCH", f"/v1/convai/tools/{tid}") for tid in fake.tools],
        ("PATCH", f"/v1/convai/agents/{agent_id}"),
    ]
    assert set(fake.tools) == set(tools_before)
    assert list(fake.agents) == [agent_id]
    assert fake.agents[agent_id]["platform_settings"]["auth"]["enable_auth"] is True
    assert "Agent updated" in capsys.readouterr().out


def test_existing_unrelated_tools_are_untouched(script: ModuleType) -> None:
    fake = FakeElevenLabs()
    other = fake.add_tool("someone_elses_tool")
    assert run(script, fake, []) == 0
    assert fake.tools[other] == {"name": "someone_elses_tool", "type": "client"}
    assert not any(path.endswith(other) for _, path in fake.calls())


def test_paginated_tool_list_finds_tools_on_later_pages(script: ModuleType) -> None:
    fake = FakeElevenLabs(page_size=2)
    for i in range(5):
        fake.add_tool(f"filler_{i}")
    existing = {name: fake.add_tool(name) for name in TOOL_NAMES}
    assert run(script, fake, []) == 0

    list_calls = [r for r in fake.requests if r.method == "GET"]
    assert len(list_calls) == 4  # 7 tools at 2 per page
    assert [r.url.params.get("cursor") for r in list_calls] == [None, "2", "4", "6"]
    assert ("POST", "/v1/convai/tools") not in fake.calls()
    for tool_id in existing.values():
        assert ("PATCH", f"/v1/convai/tools/{tool_id}") in fake.calls()
    assert len(fake.tools) == 5 + len(TOOL_NAMES)


def test_duplicate_tool_names_refused(
    script: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeElevenLabs()
    fake.add_tool(TOOL_NAMES[0])
    fake.add_tool(TOOL_NAMES[0])
    assert run(script, fake, []) == 1
    assert fake.calls() == [("GET", "/v1/convai/tools")]
    assert f"several workspace tools are named {TOOL_NAMES[0]}" in capsys.readouterr().err
    assert fake.agents == {}


# --- allow hosts ---------------------------------------------------------------------


def test_allow_host_sets_allowlist(script: ModuleType) -> None:
    fake = FakeElevenLabs()
    argv = ["--allow-host", "localhost", "--allow-host", "Demo.Example.com:8000"]
    assert run(script, fake, argv) == 0
    [agent] = fake.agents.values()
    assert agent["platform_settings"]["auth"] == {
        "enable_auth": True,
        "allowlist": [{"hostname": "localhost"}, {"hostname": "demo.example.com:8000"}],
    }


def test_update_without_allow_host_leaves_allowlist_out(script: ModuleType) -> None:
    fake = FakeElevenLabs()
    assert run(script, fake, ["--allow-host", "localhost"]) == 0
    [agent_id] = fake.agents
    assert run(script, fake, [], settings(elevenlabs_agent_id=agent_id)) == 0
    assert fake.agents[agent_id]["platform_settings"]["auth"] == {"enable_auth": True}


@pytest.mark.parametrize("host", ["http://localhost", "evil host", "a/b", "*.x.com", ""])
def test_bad_allow_host_exits_2_without_network(
    script: ModuleType, host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeElevenLabs()
    assert run(script, fake, ["--allow-host", host]) == 2
    assert fake.requests == []
    assert "--allow-host" in capsys.readouterr().err


def test_bad_config_file_exits_2(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "agent.json"
    bad.write_text(json.dumps({"name": "x"}))
    fake = FakeElevenLabs()
    assert run(script, fake, ["--config", str(bad)]) == 2
    assert fake.requests == []
    assert "invalid agent.json" in capsys.readouterr().err
    missing = tmp_path / "nope.json"
    assert run(script, fake, ["--config", str(missing)]) == 2


# --- API errors ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fail", "expected_detail"),
    [
        (
            ("GET", "/v1/convai/tools", 401, {"detail": {"message": f"invalid key {API_KEY}"}}),
            "elevenlabs: invalid key [redacted]",
        ),
        (
            ("POST", "/v1/convai/tools", 422, {"detail": [{"msg": "name taken"}]}),
            "elevenlabs: name taken",
        ),
        (
            ("POST", "/v1/convai/agents/create", 403, {"detail": "missing convai_write"}),
            "elevenlabs: missing convai_write",
        ),
        (("POST", "/v1/convai/agents/create", 500, ["oops"]), None),
    ],
)
def test_api_error_exits_1_with_detail_and_redacted_key(
    script: ModuleType,
    capsys: pytest.CaptureFixture[str],
    fail: tuple[str, str, int, Any],
    expected_detail: str | None,
) -> None:
    fake = FakeElevenLabs(fail=fail)
    assert run(script, fake, []) == 1
    captured = capsys.readouterr()
    assert f"error: elevenlabs returned status {fail[2]}" in captured.err
    if expected_detail is None:
        assert "elevenlabs:" not in captured.err
    else:
        assert expected_detail in captured.err
    assert API_KEY not in captured.out + captured.err
    assert "ELEVENLABS_AGENT_ID=" not in captured.out


def test_update_of_unknown_agent_exits_1(
    script: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeElevenLabs()
    assert run(script, fake, [], settings(elevenlabs_agent_id="agent_gone")) == 1
    assert "agent not found" in capsys.readouterr().err


def test_transport_error_exits_1(script: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout(f"timeout talking with key {API_KEY}")

    args = script.parse_args([])
    http = httpx.Client(transport=httpx.MockTransport(handler))
    assert script.run(args, settings(), http=http) == 1
    captured = capsys.readouterr()
    assert "elevenlabs request failed: ConnectTimeout" in captured.err
    assert API_KEY not in captured.out + captured.err


def test_injected_http_client_is_not_closed(script: ModuleType) -> None:
    fake = FakeElevenLabs()
    http = fake.http()
    assert script.run(script.parse_args([]), settings(), http=http) == 0
    assert not http.is_closed


# --- conversation limit and client overrides (only first_message allowed) ------------

EXPECTED_OVERRIDES = {
    "conversation_config_override": {
        "agent": {"prompt": {"prompt": False}, "first_message": True, "language": False},
        "tts": {"voice_id": False},
    }
}


def _agent_bodies(fake: FakeElevenLabs) -> list[tuple[str, dict[str, Any]]]:
    return [
        (r.method, json.loads(r.content))
        for r in fake.requests
        if r.url.path.startswith("/v1/convai/agents")
    ]


def test_create_and_update_send_duration_and_overrides(script: ModuleType) -> None:
    fake = FakeElevenLabs()
    assert run(script, fake, []) == 0
    [agent_id] = fake.agents
    assert run(script, fake, [], settings(elevenlabs_agent_id=agent_id)) == 0

    bodies = _agent_bodies(fake)
    assert [method for method, _ in bodies] == ["POST", "PATCH"]
    for _, body in bodies:
        assert body["conversation_config"]["conversation"] == {"max_duration_seconds": 180}
        assert body["platform_settings"]["overrides"] == EXPECTED_OVERRIDES
        assert body["platform_settings"]["auth"]["enable_auth"] is True
        assert body["conversation_config"]["tts"] == {"expressive_mode": False, "stability": 0.6}
        assert body["conversation_config"]["turn"] == {
            "turn_timeout": 30.0,
            "turn_eagerness": "patient",
        }
        agent = body["conversation_config"]["agent"]
        assert list(agent["prompt"]["built_in_tools"]) == ["skip_turn"]
        assert agent["max_conversation_duration_message"].startswith("We've reached")


def test_custom_duration_from_config_file(script: ModuleType, tmp_path: Path) -> None:
    data = json.loads(AGENT_JSON.read_text(encoding="utf-8"))
    data["conversation"] = {"max_duration_seconds": 60}
    custom = tmp_path / "agent.json"
    custom.write_text(json.dumps(data))
    fake = FakeElevenLabs()
    assert run(script, fake, ["--config", str(custom)]) == 0
    [(_, body)] = _agent_bodies(fake)
    assert body["conversation_config"]["conversation"] == {"max_duration_seconds": 60}


def test_out_of_range_duration_exits_2_without_network(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = json.loads(AGENT_JSON.read_text(encoding="utf-8"))
    data["conversation"] = {"max_duration_seconds": 7200}
    custom = tmp_path / "agent.json"
    custom.write_text(json.dumps(data))
    fake = FakeElevenLabs()
    assert run(script, fake, ["--config", str(custom)]) == 2
    assert fake.requests == []
    assert "conversation.max_duration_seconds" in capsys.readouterr().err


def test_dry_run_shows_duration_and_overrides(
    script: ModuleType,
    clean_env: Path,
    no_network: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert script.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"max_duration_seconds": 180' in out
    assert '"conversation_config_override"' in out
    assert '"expressive_mode": false' in out
    assert '"turn_eagerness": "patient"' in out
    assert '"turn_timeout": 30.0' in out
    assert '"system_tool_type": "skip_turn"' in out
    assert '"max_conversation_duration_message": "We\'ve reached the three-minute limit' in out


def test_create_and_update_never_send_tts_model_or_voice(script: ModuleType) -> None:
    fake = FakeElevenLabs()
    assert run(script, fake, []) == 0
    [agent_id] = fake.agents
    assert run(script, fake, [], settings(elevenlabs_agent_id=agent_id)) == 0
    bodies = _agent_bodies(fake)
    assert [method for method, _ in bodies] == ["POST", "PATCH"]
    for _, body in bodies:
        tts = body["conversation_config"]["tts"]
        assert set(tts) == {"expressive_mode", "stability"}
        assert "model_id" not in json.dumps(body)
        assert body["platform_settings"]["overrides"]["conversation_config_override"]["tts"] == {
            "voice_id": False
        }
