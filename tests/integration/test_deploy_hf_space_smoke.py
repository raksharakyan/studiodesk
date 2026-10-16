"""Minimal smoke tests for scripts/deploy_hf_space.py against a fake hub API (no network,
never the real `.env`). The full suite is written by qa-agent."""

import importlib.util
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
FAKE_TOKEN = "hf_fake-token-not-real-123456"  # noqa: S105 (test dummy)
FAKE_GROQ = "gsk_fake-groq-key-not-real-123"


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
    calls: list[tuple[str, Any]] = field(default_factory=list)
    uploaded: list[str] = field(default_factory=list)

    def whoami(self) -> dict[str, Any]:
        return {"name": "Some_User"}

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", (repo_id, kwargs)))

    def upload_folder(self, **kwargs: Any) -> None:
        folder = Path(kwargs["folder_path"])
        self.uploaded = sorted(
            p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
        )
        self.calls.append(("upload_folder", kwargs["repo_id"]))

    def add_space_variable(self, repo_id: str, key: str, value: str) -> None:
        self.calls.append(("variable", (key, value)))

    def add_space_secret(self, repo_id: str, key: str, value: str) -> None:
        self.calls.append(("secret", (key, value)))


def fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock"):
        (root / name).write_text(f"# {name}\n")
    (root / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (root / "src" / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (root / "data").mkdir()
    (root / "data" / "d.json").write_text("[]")
    (root / ".env").write_text("NEVER=staged\n")
    (root / "tests").mkdir()
    (root / "tests" / "t.py").write_text("")
    track(root)
    return root


def track(root: Path) -> None:
    """Make `root` a git checkout with every file tracked (the script stages git files)."""
    git = shutil.which("git")
    assert git is not None, "git is required for the deploy script tests"
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run([git, "-C", str(root), *args], check=True, capture_output=True)  # noqa: S603


def write_env(tmp_path: Path, extra: str = "") -> Path:
    env = tmp_path / "test.env"
    env.write_text(
        f"HF_TOKEN={FAKE_TOKEN}\nGROQ_API_KEY={FAKE_GROQ}\nANTHROPIC_API_KEY=sk-ant-x-12345678\n"
        f"GITHUB_REPO=owner/sandbox\n{extra}"
    )
    return env


def test_app_url_mapping(deploy: ModuleType) -> None:
    assert deploy.app_url("Some_User/Studio.Desk") == "https://some-user-studio-desk.hf.space"
    assert deploy.space_url("a/b") == "https://huggingface.co/spaces/a/b"


def test_dry_run_needs_no_token_and_stages_only_the_allowlist(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = fake_repo(tmp_path)
    args = deploy.parse_args(["--dry-run", "--env-file", str(tmp_path / "missing.env")])

    def no_api(token: str) -> None:
        raise AssertionError("dry run must not build an API client")

    assert deploy.run(args, environ={}, repo_root=root, api_factory=no_api) == 0
    out = capsys.readouterr().out
    assert "src/pkg/a.py" in out and "data/d.json" in out and "README.md" in out
    assert ".env" not in out.replace(".dockerignore", "")
    assert "__pycache__" not in out and "tests/" not in out
    assert "sdk: docker" in out and "app_port: 7860" in out


def test_secret_in_staged_file_fails_closed(deploy: ModuleType, tmp_path: Path) -> None:
    root = fake_repo(tmp_path)
    (root / "src" / "pkg" / "leak.py").write_text(f'KEY = "{FAKE_GROQ}"\n')
    track(root)
    args = deploy.parse_args(["--dry-run", "--env-file", str(write_env(tmp_path))])
    with pytest.raises(deploy.ConfigError, match="leak.py"):
        deploy.run(args, environ={}, repo_root=root)


def test_env_file_inside_staged_dir_fails_closed(deploy: ModuleType, tmp_path: Path) -> None:
    root = fake_repo(tmp_path)
    (root / "src" / ".env.local").write_text("X=1\n")
    track(root)
    args = deploy.parse_args(["--dry-run", "--env-file", str(tmp_path / "none.env")])
    with pytest.raises(deploy.ConfigError, match=r"\.env file"):
        deploy.run(args, environ={}, repo_root=root)


def test_deploy_syncs_names_without_hf_token_or_values_printed(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = fake_repo(tmp_path)
    hub = FakeHub()
    args = deploy.parse_args(["--sync-settings", "--yes", "--env-file", str(write_env(tmp_path))])

    code = deploy.run(args, environ={}, repo_root=root, api_factory=lambda token: hub)

    assert code == 0
    assert hub.calls[0] == (
        "create_repo",
        (
            "Some_User/studiodesk",
            {"repo_type": "space", "space_sdk": "docker", "exist_ok": True, "private": False},
        ),
    )
    secrets = dict(v for kind, v in hub.calls if kind == "secret")
    variables = dict(v for kind, v in hub.calls if kind == "variable")
    assert secrets == {"GROQ_API_KEY": FAKE_GROQ}  # no HF_TOKEN, no Anthropic for groq
    assert variables["APP_ENV"] == "prod" and variables["GITHUB_REPO"] == "owner/sandbox"
    assert "HF_TOKEN" not in variables
    assert ".env" not in hub.uploaded
    out = capsys.readouterr().out
    assert FAKE_TOKEN not in out and FAKE_GROQ not in out
    assert "https://some-user-studiodesk.hf.space" in out


def test_without_yes_a_no_answer_aborts(deploy: ModuleType, tmp_path: Path) -> None:
    root = fake_repo(tmp_path)
    hub = FakeHub()
    args = deploy.parse_args(["--env-file", str(write_env(tmp_path))])
    code = deploy.run(
        args, environ={}, repo_root=root, api_factory=lambda token: hub, ask=lambda _: "no"
    )
    assert code == deploy.EXIT_ABORTED
    assert [kind for kind, _ in hub.calls] == []


def test_dry_run_on_the_real_repo_lists_no_secrets(
    deploy: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = deploy.parse_args(["--dry-run", "--env-file", str(tmp_path / "none.env")])
    assert deploy.run(args, environ={}) == 0
    listed = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert "Dockerfile" in listed and "uv.lock" in listed
    assert not any(line.startswith((".env", "tests/", "scripts/", ".git")) for line in listed)


def test_untracked_file_under_src_aborts_with_its_path(deploy: ModuleType, tmp_path: Path) -> None:
    root = fake_repo(tmp_path)
    (root / "src" / "pkg" / "notes.txt").write_text("draft\n")
    args = deploy.parse_args(["--dry-run", "--env-file", str(tmp_path / "none.env")])
    with pytest.raises(deploy.ConfigError, match="src/pkg/notes.txt"):
        deploy.run(args, environ={}, repo_root=root)


def test_token_shapes_and_encoded_secrets_are_caught(deploy: ModuleType) -> None:
    import base64

    assert deploy.scan_bytes(b"x = 'ghp_" + b"a" * 36 + b"'", []) is not None
    assert deploy.scan_bytes(b"https://hooks.slack.com/services/T/B/x", []) is not None
    assert deploy.scan_bytes(b"gsk_your-groq-key-here", []) is None  # placeholder
    needles = sorted(deploy.encoded_forms(FAKE_GROQ))
    assert deploy.scan_bytes(base64.b64encode(FAKE_GROQ.encode()), needles) is not None
