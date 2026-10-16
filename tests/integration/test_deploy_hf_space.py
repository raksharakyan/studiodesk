"""scripts/deploy_hf_space.py: .env parsing, staging allowlist, secret scan, Space settings,
the confirm/abort path and error output, against a fake hub API.

No network and never the real `.env`: every test passes `--env-file` pointing at a
temporary dummy file and a temporary fake repository as `repo_root` (except the dry runs
over the real repo, which read only the allowlisted src/ and data/ trees).
"""

import argparse
import functools
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "deploy_hf_space.py"
TOKEN = "hf_qa-dummy-token-not-real-0123456789"  # noqa: S105 (test dummy)
GROQ = "gsk_qa-dummy-groq-key-0123456789"
GITHUB = "github_pat_qa_dummy_0123456789abcdef"
XI = "xi-qa-dummy-eleven-0123456789"
ANTHROPIC = "sk-ant-qa-dummy-0123456789"
QDRANT_KEY = "qdrant-qa-dummy-key-0123456789"
QDRANT_URL = "https://qa-cluster.cloud.qdrant.io:6333"
ALL_SECRETS = (TOKEN, GROQ, GITHUB, XI, ANTHROPIC, QDRANT_KEY)


@pytest.fixture(scope="module")
def deploy() -> ModuleType:
    spec = importlib.util.spec_from_file_location("deploy_hf_space", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeHub:
    """Records every call; `fail_on` names a method that raises `error`."""

    name: str = "Some_User"
    fail_on: str | None = None
    error: Exception | None = None
    calls: list[tuple[str, Any]] = field(default_factory=list)
    uploaded: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    def _maybe_fail(self, method: str) -> None:
        if self.fail_on == method:
            assert self.error is not None
            raise self.error

    def whoami(self) -> dict[str, Any]:
        self._maybe_fail("whoami")
        self.calls.append(("whoami", None))
        return {"name": self.name}

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self._maybe_fail("create_repo")
        self.calls.append(("create_repo", (repo_id, kwargs)))

    def upload_folder(self, **kwargs: Any) -> None:
        self._maybe_fail("upload_folder")
        folder = Path(kwargs["folder_path"])
        self.uploaded = sorted(
            p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
        )
        rest = {k: v for k, v in kwargs.items() if k != "folder_path"}
        self.calls.append(("upload_folder", rest))

    def add_space_variable(self, repo_id: str, key: str, value: str) -> None:
        self.calls.append(("variable", (repo_id, key, value)))

    def add_space_secret(self, repo_id: str, key: str, value: str) -> None:
        self.calls.append(("secret", (repo_id, key, value)))

    def factory(self, token: str) -> "FakeHub":
        self.tokens.append(token)
        return self

    def names(self, kind: str) -> list[str]:
        return [args[1] for k, args in self.calls if k == kind]


def fake_repo(tmp_path: Path) -> Path:
    """A repository with every allowlisted input plus things that must never be staged."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "README.md"):
        (root / name).write_text(f"# {name}\n")
    pkg = root / "src" / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "static").mkdir()
    (pkg / "static" / "home.js").write_text("export {};\n")
    (pkg / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (pkg / "b.pyc").write_bytes(b"\0")
    (pkg / ".DS_Store").write_bytes(b"\0")
    (root / "src" / ".mypy_cache").mkdir()
    (root / "src" / ".mypy_cache" / "x.json").write_text("{}")
    (root / "data" / "synthetic").mkdir(parents=True)
    (root / "data" / "synthetic" / "bugs.json").write_text("[]")
    (root / "data" / ".pytest_cache").mkdir()
    (root / "data" / ".pytest_cache" / "v").write_text("x")
    for junk in ("tests", "scripts", "evals", "voice", "docs", ".github", ".claude"):
        (root / junk).mkdir()
        (root / junk / "x.py").write_text("print('no')\n")
    for env in (".env", ".env.local", ".env.example"):
        (root / env).write_text("SECRET=never\n")
    track(root)
    return root


def track(root: Path) -> None:
    """Make `root` a git checkout with every file tracked (the script stages git files)."""
    git = shutil.which("git")
    assert git is not None, "git is required for the deploy script tests"
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run([git, "-C", str(root), *args], check=True, capture_output=True)  # noqa: S603


EXPECTED_STAGED = [
    ".dockerignore",
    "Dockerfile",
    "README.md",
    "data/synthetic/bugs.json",
    "pyproject.toml",
    "src/pkg/a.py",
    "src/pkg/static/home.js",
    "uv.lock",
]


def write_env(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "dummy.env"
    path.write_text(body, encoding="utf-8")
    return path


FULL_ENV = (
    f"HF_TOKEN={TOKEN}\n"
    f"GROQ_API_KEY={GROQ}\n"
    f"ANTHROPIC_API_KEY={ANTHROPIC}\n"
    f"GITHUB_TOKEN={GITHUB}\n"
    f"ELEVENLABS_API_KEY={XI}\n"
    "ELEVENLABS_AGENT_ID=agent_qa123\n"
    f"QDRANT_URL={QDRANT_URL}\n"
    f"QDRANT_API_KEY={QDRANT_KEY}\n"
    "GITHUB_REPO=owner/sandbox\n"
)


def args(deploy: ModuleType, env_file: Path, *extra: str) -> argparse.Namespace:
    namespace: argparse.Namespace = deploy.parse_args(["--env-file", str(env_file), *extra])
    return namespace


def run(
    deploy: ModuleType,
    repo: Path,
    env_file: Path,
    *extra: str,
    hub: FakeHub | None = None,
    environ: dict[str, str] | None = None,
    answer: str = "yes",
) -> tuple[int, FakeHub]:
    hub = hub or FakeHub()
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return answer

    code: int = deploy.run(
        args(deploy, env_file, *extra),
        environ=environ or {},
        repo_root=repo,
        api_factory=hub.factory,
        ask=ask,
    )
    return code, hub


def run_main(
    deploy: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    argv: list[str],
    hub: FakeHub,
    answer: str = "yes",
) -> int:
    """main() with the fake repo, fake hub and a scripted answer (process env is cleared)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    original = deploy.run
    monkeypatch.setattr(
        deploy,
        "run",
        functools.partial(original, repo_root=repo, api_factory=hub.factory, ask=lambda _: answer),
    )
    code: int = deploy.main(argv)
    return code


def no_secret_in(text: str) -> None:
    for secret in ALL_SECRETS:
        assert secret not in text, "a secret value was printed"


# =====================================================================================
# .env loader
# =====================================================================================


def test_read_dotenv_formats(deploy: ModuleType, tmp_path: Path) -> None:
    env = write_env(
        tmp_path,
        "# a comment\n"
        "\n"
        "   # indented comment\n"
        "PLAIN=value\n"
        'DOUBLE="has spaces"\n'
        "SINGLE='single quoted'\n"
        'HASH_IN_QUOTES="a # b"\n'
        "INLINE=value # trailing comment\n"
        "NOSPACE=val#hash\n"
        "export EXPORTED=yes\n"
        "  export   SPACED_EXPORT  =  padded  \n"
        "EMPTY=\n"
        'EMPTY_QUOTES=""\n'
        "EQUALS=a=b=c\n"
        "not a valid line\n"
        "1BAD=x\n"
        "DUP=first\nDUP=second\n",
    )
    assert deploy.read_dotenv(env) == {
        "PLAIN": "value",
        "DOUBLE": "has spaces",
        "SINGLE": "single quoted",
        "HASH_IN_QUOTES": "a # b",
        "INLINE": "value",
        "NOSPACE": "val#hash",
        "EXPORTED": "yes",
        "SPACED_EXPORT": "padded",
        "EMPTY": "",
        "EMPTY_QUOTES": "",
        "EQUALS": "a=b=c",
        "DUP": "second",
    }


def test_read_dotenv_missing_file_is_empty(deploy: ModuleType, tmp_path: Path) -> None:
    assert deploy.read_dotenv(tmp_path / "nope.env") == {}


@pytest.mark.parametrize("line", ['KEY="quoted" # note', "KEY='quoted'  # note"])
def test_read_dotenv_quoted_value_with_inline_comment(
    deploy: ModuleType, tmp_path: Path, line: str
) -> None:
    assert deploy.read_dotenv(write_env(tmp_path, line + "\n")) == {"KEY": "quoted"}


def test_secret_scan_catches_quoted_secret_with_comment(deploy: ModuleType, tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / "src" / "pkg" / "leak.py").write_text(f"KEY = '{GROQ}'\n")
    track(repo)
    env = write_env(tmp_path, f'GROQ_API_KEY="{GROQ}" # my key\n')
    with pytest.raises(deploy.ConfigError):
        run(deploy, repo, env, "--dry-run")


# =====================================================================================
# Staging allowlist and README
# =====================================================================================


def test_dry_run_stages_exactly_the_allowlist(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = fake_repo(tmp_path)
    code, hub = run(deploy, repo, write_env(tmp_path, ""), "--dry-run")
    out = capsys.readouterr().out
    assert code == 0
    assert hub.calls == [] and hub.tokens == []  # no hub client was even created
    listed = [line.strip() for line in out.splitlines() if line.startswith("  ")]
    assert listed == EXPECTED_STAGED
    assert "Space: <you>/studiodesk (public, Docker SDK)" in out
    assert "Space settings: unchanged" in out


def test_dry_run_needs_no_token_or_network(
    deploy: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(token: str) -> Any:
        raise AssertionError("hub client created during a dry run")

    code = deploy.run(
        args(deploy, write_env(tmp_path, ""), "--dry-run"),
        environ={},
        repo_root=fake_repo(tmp_path),
        api_factory=boom,
        ask=lambda _: "no",
    )
    assert code == 0


def test_real_repo_dry_run_lists_only_allowed_paths(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = deploy.run(
        args(deploy, write_env(tmp_path, ""), "--dry-run"),
        environ={},
        api_factory=lambda token: pytest.fail("no hub in a dry run"),
    )
    assert code == 0
    listed = [
        line.strip() for line in capsys.readouterr().out.splitlines() if line.startswith("  ")
    ]
    assert listed
    top = {"Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "README.md"}
    for path in listed:
        first = path.split("/")[0]
        assert path in top or first in ("src", "data"), path
        parts = path.split("/")
        assert not any(p.startswith(".env") for p in parts), path
        assert "__pycache__" not in parts and not path.endswith((".pyc", ".pyo")), path
    for forbidden in ("tests/", "scripts/", "evals/", "voice/", ".github/", ".claude/"):
        assert not any(p.startswith(forbidden) for p in listed), forbidden
    assert "src/studiodesk/main.py" in listed
    assert "src/studiodesk/static/index.html" in listed


def front_matter(readme: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", readme, re.S)
    assert match is not None, "README has no front matter"
    pairs = (line.split(":", 1) for line in match.group(1).splitlines())
    return {k.strip(): v.strip() for k, v in pairs}


def test_generated_readme_front_matter(deploy: ModuleType) -> None:
    readme = deploy.space_readme("https://github.com/o/r")
    meta = front_matter(readme)
    assert meta["sdk"] == "docker"
    assert meta["app_port"] == "7860"
    assert meta["title"] == "StudioDesk"
    assert meta["pinned"] == "false"
    assert "Source code, docs and eval reports: https://github.com/o/r" in readme


def test_staged_readme_is_the_generated_one(deploy: ModuleType, tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    env = write_env(tmp_path, f"HF_TOKEN={TOKEN}\nSOURCE_REPO_URL=https://github.com/me/fork\n")
    captured: dict[str, str] = {}

    class ReadingHub(FakeHub):
        def upload_folder(self, **kwargs: Any) -> None:
            captured["readme"] = (Path(kwargs["folder_path"]) / "README.md").read_text()
            super().upload_folder(**kwargs)

    code, _ = run(deploy, repo, env, "--yes", hub=ReadingHub())
    assert code == 0
    assert front_matter(captured["readme"])["app_port"] == "7860"
    assert "https://github.com/me/fork" in captured["readme"]
    assert "# README.md" not in captured["readme"]  # the repo's own README is not staged


@pytest.mark.parametrize(
    "value",
    ["http://github.com/x/y", "https://u:p@github.com/x/y", "https://github.com/x?y=1", "ftp://x"],
)
def test_invalid_source_repo_url_rejected(deploy: ModuleType, tmp_path: Path, value: str) -> None:
    with pytest.raises(deploy.ConfigError, match="SOURCE_REPO_URL"):
        run(
            deploy,
            fake_repo(tmp_path),
            write_env(tmp_path, f"SOURCE_REPO_URL={value}\n"),
            "--dry-run",
        )


def test_symlink_in_staged_tree_refused(deploy: ModuleType, tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (repo / "src" / "pkg" / "link.txt").symlink_to(outside)
    track(repo)
    with pytest.raises(deploy.ConfigError, match="symlink"):
        run(deploy, repo, write_env(tmp_path, ""), "--dry-run")


@pytest.mark.parametrize("name", [".env", ".env.production", ".envrc"])
def test_env_file_inside_src_refused(deploy: ModuleType, tmp_path: Path, name: str) -> None:
    repo = fake_repo(tmp_path)
    (repo / "src" / "pkg" / name).write_text("X=1\n")
    track(repo)
    with pytest.raises(deploy.ConfigError, match=r"\.env file in staging"):
        run(deploy, repo, write_env(tmp_path, ""), "--dry-run")


@pytest.mark.parametrize("missing", ["Dockerfile", "uv.lock", "src", "data"])
def test_missing_inputs_refused(deploy: ModuleType, tmp_path: Path, missing: str) -> None:
    repo = fake_repo(tmp_path)
    target = repo / missing
    if target.is_dir():
        import shutil

        shutil.rmtree(target)
    else:
        target.unlink()
    with pytest.raises(deploy.ConfigError):
        run(deploy, repo, write_env(tmp_path, ""), "--dry-run")


# =====================================================================================
# Secret scan
# =====================================================================================


@pytest.mark.parametrize(
    ("env_line", "secret", "where"),
    [
        (f"GROQ_API_KEY={GROQ}", GROQ, "src/pkg/a.py"),
        (f"GITHUB_TOKEN={GITHUB}", GITHUB, "data/synthetic/bugs.json"),
        (f"ELEVENLABS_API_KEY={XI}", XI, "Dockerfile"),
        (f"QDRANT_URL={QDRANT_URL}", QDRANT_URL, "pyproject.toml"),
        ("MY_DB_PASSWORD=hunter2-hunter2", "hunter2-hunter2", "src/pkg/static/home.js"),
        (
            "CUSTOM_WEBHOOK=https://example.com/hook/abc123",
            "https://example.com/hook/abc123",
            "uv.lock",
        ),
        ("SOME_SECRET=s3cr3t-value-xyz", "s3cr3t-value-xyz", "src/pkg/a.py"),
        (f"HF_TOKEN={TOKEN}", TOKEN, "src/pkg/a.py"),
    ],
)
def test_secret_in_staged_file_aborts_naming_file_not_value(
    deploy: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env_line: str,
    secret: str,
    where: str,
) -> None:
    repo = fake_repo(tmp_path)
    target = repo / where
    target.write_text(target.read_text() + f"\nleaked = '{secret}'\n")
    env = write_env(tmp_path, env_line + "\n")
    with pytest.raises(deploy.ConfigError) as info:
        run(deploy, repo, env, "--dry-run")
    assert where in str(info.value)
    assert secret not in str(info.value)

    hub = FakeHub()
    code = run_main(deploy, monkeypatch, repo, ["--env-file", str(env), "--dry-run"], hub)
    captured = capsys.readouterr()
    assert code == 2
    assert where in captured.err
    assert secret not in captured.out + captured.err
    assert hub.calls == []


def test_hf_token_from_environment_is_scanned(deploy: ModuleType, tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / "src" / "pkg" / "a.py").write_text(f"T = '{TOKEN}'\n")
    with pytest.raises(deploy.ConfigError, match="src/pkg/a.py"):
        run(deploy, repo, write_env(tmp_path, ""), "--dry-run", environ={"HF_TOKEN": TOKEN})


@pytest.mark.parametrize(
    ("env_line", "content"),
    [
        ("APP_ENV=prod", "prod"),  # not a secret-like name
        ("GITHUB_REPO=owner/sandbox", "owner/sandbox"),
        ("GROQ_API_KEY=short", "short"),  # below the scan length
        ("QDRANT_API_KEY=your-qdrant-api-key", "your-qdrant-api-key"),  # placeholder
        ("GROQ_API_KEY=gsk_your-groq-key-here", "gsk_your-groq-key-here"),
        ("GROQ_API_KEY=", ""),
    ],
)
def test_non_secrets_do_not_abort(
    deploy: ModuleType, tmp_path: Path, env_line: str, content: str
) -> None:
    repo = fake_repo(tmp_path)
    (repo / "src" / "pkg" / "a.py").write_text(f"x = '{content}'\n")
    code, _ = run(deploy, repo, write_env(tmp_path, env_line + "\n"), "--dry-run")
    assert code == 0


def test_secret_values_list(deploy: ModuleType) -> None:
    dotenv = {
        "GROQ_API_KEY": GROQ,
        "PLAIN": "not-a-secret-value",
        "X_TOKEN": "short",
        "QDRANT_API_KEY": "your-qdrant-api-key",
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T/B/x",
    }
    values = deploy.secret_values(dotenv, {"HF_TOKEN": TOKEN, "OTHER_TOKEN": "env-only-ignored"})
    assert values == sorted([GROQ, "https://hooks.slack.com/services/T/B/x", TOKEN])


# =====================================================================================
# Space settings
# =====================================================================================


def test_space_settings_groq(deploy: ModuleType) -> None:
    settings = deploy.space_settings(
        {
            "HF_TOKEN": TOKEN,
            "GROQ_API_KEY": GROQ,
            "ANTHROPIC_API_KEY": ANTHROPIC,
            "GITHUB_TOKEN": GITHUB,
            "QDRANT_URL": QDRANT_URL,
            "GITHUB_REPO": "owner/sandbox",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T/B/x",
            "UNRELATED_TOKEN": "zzzzzzzzzzzz",
        },
        [],
    )
    assert settings.variables == {
        "APP_ENV": "prod",
        "DEBUG_LOG_CLIENT_IP": "false",
        "ACTIONS_ENABLED": "true",
        "ACTIONS_MAX_PER_DAY": "5",
        "VOICE_ENABLED": "true",
        "VOICE_MAX_SESSIONS_PER_DAY": "5",
        "LLM_PROVIDER": "groq",
        "QDRANT_COLLECTION": "studiodesk",
        "GITHUB_REPO": "owner/sandbox",
    }
    assert settings.secrets == {
        "GROQ_API_KEY": GROQ,
        "GITHUB_TOKEN": GITHUB,
        "QDRANT_URL": QDRANT_URL,
    }


def test_anthropic_secret_only_with_anthropic_provider(deploy: ModuleType) -> None:
    base = {"GROQ_API_KEY": GROQ, "ANTHROPIC_API_KEY": ANTHROPIC}
    groq = deploy.space_settings(base, [])
    anthropic = deploy.space_settings({**base, "LLM_PROVIDER": "anthropic"}, [])
    assert "ANTHROPIC_API_KEY" not in groq.secrets
    assert anthropic.secrets["ANTHROPIC_API_KEY"] == ANTHROPIC
    assert anthropic.variables["LLM_PROVIDER"] == "anthropic"


def test_placeholders_are_skipped(deploy: ModuleType) -> None:
    settings = deploy.space_settings(
        {
            "GROQ_API_KEY": "gsk_your-groq-key-here",
            "QDRANT_URL": "https://your-cluster.cloud.qdrant.io:6333",
            "QDRANT_API_KEY": "your-qdrant-api-key",
            "GITHUB_TOKEN": "github_pat_your-token-here",
            "ELEVENLABS_API_KEY": "your-elevenlabs-api-key",
            "ELEVENLABS_AGENT_ID": "   ",
        },
        [],
    )
    assert settings.secrets == {}


def test_hf_token_never_a_setting(deploy: ModuleType) -> None:
    settings = deploy.space_settings({"HF_TOKEN": TOKEN, "GROQ_API_KEY": GROQ}, [])
    assert "HF_TOKEN" not in settings.secrets and "HF_TOKEN" not in settings.variables
    assert TOKEN not in settings.secrets.values() and TOKEN not in settings.variables.values()


def test_hf_token_guard_fails_closed(deploy: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deploy, "SPACE_SECRET_NAMES", (*deploy.SPACE_SECRET_NAMES, "HF_TOKEN"))
    with pytest.raises(deploy.ConfigError, match="HF_TOKEN"):
        deploy.space_settings({"HF_TOKEN": TOKEN}, [])


@pytest.mark.parametrize(
    ("dotenv", "match"),
    [
        ({"GITHUB_REPO": "not a repo"}, "GITHUB_REPO"),
        ({"GITHUB_REPO": "owner/repo/extra"}, "GITHUB_REPO"),
        ({"GITHUB_REPO": "https://github.com/o/r"}, "GITHUB_REPO"),
        ({"QDRANT_COLLECTION": "bad name!"}, "QDRANT_COLLECTION"),
        ({"QDRANT_COLLECTION": "a" * 256}, "QDRANT_COLLECTION"),
        ({"LLM_PROVIDER": "openai"}, "LLM_PROVIDER"),
    ],
)
def test_invalid_settings_rejected(deploy: ModuleType, dotenv: dict[str, str], match: str) -> None:
    with pytest.raises(deploy.ConfigError, match=match):
        deploy.space_settings(dotenv, [])


@pytest.mark.parametrize("proxy", ["0.0.0.0/0", "::/0", "not-an-ip", "10.0.0.0/33", "1.2.3"])
def test_invalid_or_catch_all_proxy_rejected(deploy: ModuleType, proxy: str) -> None:
    with pytest.raises(deploy.ConfigError, match="trusted-proxy"):
        deploy.space_settings({}, [proxy])


def test_valid_proxies_normalised(deploy: ModuleType) -> None:
    settings = deploy.space_settings({}, ["10.0.0.5/8", "192.168.1.1, fd00::/8", ""])
    assert settings.variables["TRUSTED_PROXY_IPS"] == "10.0.0.0/8,192.168.1.1/32,fd00::/8"


def test_trusted_proxy_needs_sync_settings(deploy: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(deploy.ConfigError, match="--sync-settings"):
        run(
            deploy,
            fake_repo(tmp_path),
            write_env(tmp_path, ""),
            "--dry-run",
            "--trusted-proxy",
            "10.0.0.1",
        )


def test_dry_run_with_sync_prints_names_only(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run(
        deploy, fake_repo(tmp_path), write_env(tmp_path, FULL_ENV), "--dry-run", "--sync-settings"
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Space secrets (values from .env, not shown): " in out
    assert "GROQ_API_KEY" in out and "QDRANT_URL" in out
    no_secret_in(out)
    assert QDRANT_URL not in out
    assert "HF_TOKEN" not in out.split("Space variables:")[1]


# =====================================================================================
# Deploy, confirm and abort
# =====================================================================================


def test_full_deploy_calls_and_outputs(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, hub = run(deploy, fake_repo(tmp_path), write_env(tmp_path, FULL_ENV), "--sync-settings")
    captured = capsys.readouterr()
    assert code == 0
    assert hub.tokens == [TOKEN]
    kinds = [kind for kind, _ in hub.calls]
    assert kinds[:3] == ["whoami", "create_repo", "upload_folder"]
    assert hub.calls[1] == (
        "create_repo",
        (
            "Some_User/studiodesk",
            {"repo_type": "space", "space_sdk": "docker", "exist_ok": True, "private": False},
        ),
    )
    assert hub.calls[2][1] == {
        "repo_id": "Some_User/studiodesk",
        "repo_type": "space",
        "commit_message": "Deploy StudioDesk",
        "delete_patterns": ["*"],
    }
    assert hub.uploaded == EXPECTED_STAGED
    assert "HF_TOKEN" not in hub.names("secret") + hub.names("variable")
    assert set(hub.names("secret")) == {
        "GROQ_API_KEY",
        "GITHUB_TOKEN",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_AGENT_ID",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    }
    assert all(
        args[0] == "Some_User/studiodesk" for k, args in hub.calls if k in ("secret", "variable")
    )
    assert "Space: https://huggingface.co/spaces/Some_User/studiodesk" in captured.out
    assert "App:   https://some-user-studiodesk.hf.space" in captured.out
    no_secret_in(captured.out + captured.err)


def test_without_sync_settings_nothing_is_set(deploy: ModuleType, tmp_path: Path) -> None:
    code, hub = run(
        deploy, fake_repo(tmp_path), write_env(tmp_path, FULL_ENV), "--space", "me/demo"
    )
    assert code == 0
    assert hub.names("secret") == [] and hub.names("variable") == []
    assert "whoami" not in [k for k, _ in hub.calls]  # --space given: no lookup


@pytest.mark.parametrize("answer", ["no", "", "y", "YES please", "n"])
def test_anything_but_yes_aborts_before_upload(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], answer: str
) -> None:
    code, hub = run(
        deploy, fake_repo(tmp_path), write_env(tmp_path, FULL_ENV), "--sync-settings", answer=answer
    )
    assert code == 3
    assert [k for k, _ in hub.calls] == ["whoami"]
    assert "aborted: nothing was uploaded" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["yes", " YES ", "Yes"])
def test_yes_answers_proceed(deploy: ModuleType, tmp_path: Path, answer: str) -> None:
    code, hub = run(deploy, fake_repo(tmp_path), write_env(tmp_path, FULL_ENV), answer=answer)
    assert code == 0
    assert "upload_folder" in [k for k, _ in hub.calls]


def test_missing_token_is_a_config_error(deploy: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(deploy.ConfigError, match="HF_TOKEN is not set"):
        run(deploy, fake_repo(tmp_path), write_env(tmp_path, f"GROQ_API_KEY={GROQ}\n"))


def test_token_from_environment_wins(deploy: ModuleType, tmp_path: Path) -> None:
    env_token = "hf_from-the-environment-0123456789"  # noqa: S105 (test dummy)
    _, hub = run(
        deploy,
        fake_repo(tmp_path),
        write_env(tmp_path, FULL_ENV),
        "--yes",
        environ={"HF_TOKEN": env_token},
    )
    assert hub.tokens == [env_token]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Some_User", "Some_User/studiodesk"), ("a.b-c", "a.b-c/studiodesk")],
)
def test_default_space_from_whoami(
    deploy: ModuleType, tmp_path: Path, name: str, expected: str
) -> None:
    _, hub = run(
        deploy, fake_repo(tmp_path), write_env(tmp_path, FULL_ENV), "--yes", hub=FakeHub(name=name)
    )
    assert hub.calls[1][1][0] == expected


@pytest.mark.parametrize("name", ["../evil", "has space", ""])
def test_unsafe_whoami_name_rejected(deploy: ModuleType, tmp_path: Path, name: str) -> None:
    with pytest.raises(deploy.ConfigError):
        run(
            deploy,
            fake_repo(tmp_path),
            write_env(tmp_path, FULL_ENV),
            "--yes",
            hub=FakeHub(name=name),
        )


@pytest.mark.parametrize(
    "space", ["noslash", "a/b/c", "../x", "a/..", "/b", "a/", "-a/b", "a/b c", "a" * 97 + "/b"]
)
def test_invalid_space_id_rejected(deploy: ModuleType, tmp_path: Path, space: str) -> None:
    with pytest.raises(deploy.ConfigError, match="owner/name"):
        run(deploy, fake_repo(tmp_path), write_env(tmp_path, ""), "--dry-run", f"--space={space}")


@pytest.mark.parametrize(
    ("space", "url"),
    [
        ("Some_User/Studio.Desk", "https://some-user-studio-desk.hf.space"),
        ("me/studiodesk", "https://me-studiodesk.hf.space"),
        ("ORG/My_App.v2", "https://org-my-app-v2.hf.space"),
        ("a-b/c-d", "https://a-b-c-d.hf.space"),
    ],
)
def test_app_url_subdomain_mapping(deploy: ModuleType, space: str, url: str) -> None:
    assert deploy.app_url(space) == url
    assert deploy.space_url(space) == f"https://huggingface.co/spaces/{space}"


# =====================================================================================
# Error paths never print the token
# =====================================================================================


class FakeHTTPError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.response = type("R", (), {"status_code": status})()


@pytest.mark.parametrize("method", ["whoami", "create_repo", "upload_folder"])
@pytest.mark.parametrize(
    ("error", "suffix"),
    [
        (FakeHTTPError(f"401 for token {TOKEN} and {GROQ}", 401), " (HTTP 401)"),
        (RuntimeError(f"boom {TOKEN}"), ""),
        (OSError(f"network down, Authorization: Bearer {TOKEN}"), ""),
    ],
)
def test_hub_errors_print_type_only(
    deploy: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    method: str,
    error: Exception,
    suffix: str,
) -> None:
    hub = FakeHub(fail_on=method, error=error)
    env = write_env(tmp_path, FULL_ENV)
    code = run_main(
        deploy, monkeypatch, fake_repo(tmp_path), ["--env-file", str(env), "--yes"], hub
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == (
        f"error: Hugging Face request failed: {type(error).__name__}{suffix}"
    )
    no_secret_in(captured.out + captured.err)


def test_config_errors_print_message_without_values(
    deploy: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = write_env(tmp_path, FULL_ENV + f"GITHUB_REPO=bad repo {GROQ}\n")
    code = run_main(
        deploy,
        monkeypatch,
        fake_repo(tmp_path),
        ["--env-file", str(env), "--sync-settings"],
        FakeHub(),
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.strip() == "error: GITHUB_REPO in .env is not valid"
    no_secret_in(captured.out + captured.err)


def test_main_never_reads_the_real_env_when_env_file_given(
    deploy: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    original = deploy.read_dotenv

    def spy(path: Path) -> dict[str, str]:
        seen.append(Path(path))
        return original(path)  # type: ignore[no-any-return]

    monkeypatch.setattr(deploy, "read_dotenv", spy)
    env = write_env(tmp_path, "")
    run_main(
        deploy, monkeypatch, fake_repo(tmp_path), ["--env-file", str(env), "--dry-run"], FakeHub()
    )
    assert seen == [env]
    assert REPO_ROOT / ".env" not in seen
    assert os.environ.get("HF_TOKEN") is None
