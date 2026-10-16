"""Deploy StudioDesk to a public Hugging Face Docker Space (idempotent).

Builds a staging directory in a temporary folder from an explicit allowlist (Dockerfile,
.dockerignore, pyproject.toml, uv.lock, src/ and data/ without caches, plus a generated
Space README with the YAML front matter HF needs), refuses to continue if any staged file
is named `.env*`, is a symlink, or contains the value of a secret from `.env`, then mirrors
it to the Space with `upload_folder`.

`--sync-settings` also sets the Space's non-secret *variables* (prod switches and caps)
and its *secrets* from a fixed allowlist read only from `.env` (never from the shell
environment, which may hold unrelated tokens such as a personal GITHUB_TOKEN). Only names
are printed, never values.

HF_TOKEN (write scope) is read from the environment or `.env`, used only for the upload,
never printed and never stored as a Space secret.

    uv run python scripts/deploy_hf_space.py --dry-run           # list files + README, no network
    uv run python scripts/deploy_hf_space.py [--space owner/name] [--sync-settings]
        [--trusted-proxy CIDR ...] [--yes]

Exit codes: 0 ok, 1 Hugging Face API error, 2 configuration error, 3 aborted.
"""

import argparse
import ipaddress
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from studiodesk.config import (
    DEFAULT_SOURCE_REPO_URL,
    GITHUB_REPO_PATTERN,
    PLACEHOLDER_VALUES,
    PUBLIC_URL_PATTERN,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_SPACE_NAME = "studiodesk"
HF_SPACES_URL = "https://huggingface.co/spaces"
APP_PORT = 7860

EXIT_API_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_ABORTED = 3

# What the Space needs to build the image; nothing else is ever staged.
STAGED_FILES = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock")
STAGED_DIRS = ("src", "data")
SKIPPED_DIR_NAMES = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})
SKIPPED_SUFFIXES = (".pyc", ".pyo")
SKIPPED_FILE_NAMES = frozenset({".DS_Store"})
FORBIDDEN_NAME_PREFIX = ".env"

TOKEN_NAME = "HF_TOKEN"  # noqa: S105 (an env var name, not a secret)
# Space secrets, read only from `.env`. HF_TOKEN is never one of them.
SPACE_SECRET_NAMES = (
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "GROQ_API_KEY",
    "GITHUB_TOKEN",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_AGENT_ID",
)
ANTHROPIC_SECRET_NAME = "ANTHROPIC_API_KEY"  # noqa: S105 (an env var name, not a secret)
# Fixed prod switches and caps for the public demo.
FIXED_VARIABLES = {
    "APP_ENV": "prod",
    "ACTIONS_ENABLED": "true",
    "ACTIONS_MAX_PER_DAY": "5",
    "VOICE_ENABLED": "true",
    "VOICE_MAX_SESSIONS_PER_DAY": "5",
}
LLM_PROVIDERS = frozenset({"groq", "anthropic"})
DEFAULT_COLLECTION = "studiodesk"
COLLECTION_PATTERN = r"^[A-Za-z0-9_-]{1,255}$"
# Names whose `.env` values must never appear in a staged file.
SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|WEBHOOK)", re.IGNORECASE)
MIN_SCANNED_SECRET_CHARS = 8
SPACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_DOTENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

README_TEMPLATE = """---
title: StudioDesk
emoji: 🛰️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: {port}
pinned: false
license: mit
short_description: AI support and QA agent for game studios (demo)
---

# StudioDesk

AI support and QA agent for game studios: cited answers over bug reports, crash logs and
patch notes, duplicate bug detection, and GitHub issue filing after an explicit confirm
step, with an optional voice mode.

The demo data is a fictional game, *Starfall Outpost*. Issues filed from this demo go to
a public sandbox repository, and filing and voice sessions have small daily caps.

Source code, docs and eval reports: {source_repo_url}
"""


class ConfigError(Exception):
    """Invalid input or an unsafe staging directory (exit code 2)."""


class HubApi(Protocol):
    """The subset of `huggingface_hub.HfApi` this script uses (fakeable in tests)."""

    def whoami(self) -> dict[str, Any]:
        """Return the token owner's account info."""
        ...

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool | None = None,
        repo_type: str | None = None,
        exist_ok: bool = False,
        space_sdk: str | None = None,
    ) -> Any:
        """Create the repository (no-op with `exist_ok=True` if it exists)."""
        ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        commit_message: str | None = None,
        repo_type: str | None = None,
        delete_patterns: list[str] | str | None = None,
    ) -> Any:
        """Commit a local folder to the repository."""
        ...

    def add_space_variable(self, repo_id: str, key: str, value: str) -> Any:
        """Set a public Space variable."""
        ...

    def add_space_secret(self, repo_id: str, key: str, value: str) -> Any:
        """Set a private Space secret."""
        ...


@dataclass(frozen=True)
class SpaceSettings:
    """Variables (non-secret) and secrets to set on the Space. Values are never printed."""

    variables: dict[str, str]
    secrets: dict[str, str]


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a `.env` file (KEY=VALUE, optional quotes, `export`, # comments).

    A missing file yields an empty dict. Values are returned, never printed.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _DOTENV_LINE_RE.match(line)
        if match is None:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def is_set(value: str | None) -> bool:
    """True for a non-empty value that is not an `.env.example` placeholder."""
    return bool(value and value.strip() and value.strip() not in PLACEHOLDER_VALUES)


def resolve_token(environ: Mapping[str, str], dotenv: Mapping[str, str]) -> str | None:
    """HF_TOKEN from the environment, else from `.env` (None if unset)."""
    for source in (environ, dotenv):
        value = source.get(TOKEN_NAME, "").strip()
        if value:
            return value
    return None


def validate_space_id(space: str) -> str:
    """Return `space` if it looks like `owner/name`, else raise ConfigError."""
    if not SPACE_ID_RE.fullmatch(space) or ".." in space:
        raise ConfigError("--space must look like owner/name")
    return space


def space_url(space: str) -> str:
    """The Space's page on huggingface.co."""
    return f"{HF_SPACES_URL}/{space}"


def app_url(space: str) -> str:
    """The Space's app URL: `https://<owner>-<name>.hf.space`, mapped the way HF does
    (lowercase; `/`, `_` and `.` become `-`)."""
    subdomain = re.sub(r"[/_.]", "-", space.lower())
    return f"https://{subdomain}.hf.space"


def space_readme(source_repo_url: str) -> str:
    """The Space README: HF front matter (docker SDK, port 7860) and a short description."""
    return README_TEMPLATE.format(port=APP_PORT, source_repo_url=source_repo_url)


def resolve_source_repo_url(dotenv: Mapping[str, str]) -> str:
    """SOURCE_REPO_URL from `.env` if set and valid, else the default GitHub repo."""
    value = dotenv.get("SOURCE_REPO_URL", "").strip()
    if not value:
        return DEFAULT_SOURCE_REPO_URL
    if not re.fullmatch(PUBLIC_URL_PATTERN, value):
        raise ConfigError("SOURCE_REPO_URL must be a plain https URL")
    return value


def _skipped(path: Path) -> bool:
    """Caches, bytecode and OS litter are never staged."""
    return (
        path.name in SKIPPED_FILE_NAMES
        or path.name in SKIPPED_DIR_NAMES
        or path.suffix in SKIPPED_SUFFIXES
    )


def _copy_tree(source: Path, target: Path) -> None:
    """Copy regular files under `source`, skipping caches; refuse symlinks."""
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(_skipped(Path(part)) for part in relative.parts):
            continue
        if path.is_symlink():
            raise ConfigError(f"refusing to stage a symlink: {source.name}/{relative}")
        if path.is_dir():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


def build_staging(repo_root: Path, staging: Path, readme: str) -> None:
    """Fill `staging` with exactly the allowlisted files and the generated README."""
    for name in STAGED_FILES:
        source = repo_root / name
        if not source.is_file() or source.is_symlink():
            raise ConfigError(f"missing or not a regular file: {name}")
        shutil.copyfile(source, staging / name)
    for name in STAGED_DIRS:
        source = repo_root / name
        if not source.is_dir() or source.is_symlink():
            raise ConfigError(f"missing directory: {name}")
        _copy_tree(source, staging / name)
    (staging / "README.md").write_text(readme, encoding="utf-8")


def staged_files(staging: Path) -> list[str]:
    """Relative POSIX paths of every file in `staging`, sorted."""
    return sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())


def secret_values(dotenv: Mapping[str, str], environ: Mapping[str, str]) -> list[str]:
    """Values that must never be uploaded: every secret-like `.env` entry, the Space
    secrets, and HF_TOKEN from either source (placeholders and short values excluded)."""
    names = {k for k in dotenv if SECRET_NAME_RE.search(k)}
    names |= {*SPACE_SECRET_NAMES, ANTHROPIC_SECRET_NAME, "SLACK_WEBHOOK_URL", TOKEN_NAME}
    values = {dotenv.get(name, "").strip() for name in names}
    values.add(environ.get(TOKEN_NAME, "").strip())
    return sorted(v for v in values if is_set(v) and len(v) >= MIN_SCANNED_SECRET_CHARS)


def check_staging(staging: Path, secrets: Sequence[str]) -> None:
    """Fail closed if any staged path is `.env*`, a symlink, or contains a secret value.

    Error messages name the file, never the secret.
    """
    encoded = [s.encode("utf-8") for s in secrets]
    for path in sorted(staging.rglob("*")):
        relative = path.relative_to(staging).as_posix()
        if path.is_symlink():
            raise ConfigError(f"symlink in staging: {relative}")
        if path.name.startswith(FORBIDDEN_NAME_PREFIX):
            raise ConfigError(f".env file in staging: {relative}")
        if path.is_file():
            data = path.read_bytes()
            if any(secret in data for secret in encoded):
                raise ConfigError(f"a secret value from .env appears in {relative}")


def _validated(name: str, value: str, pattern: str) -> str:
    """Return `value` if it matches `pattern`, else raise ConfigError naming the setting."""
    if not re.fullmatch(pattern, value):
        raise ConfigError(f"{name} in .env is not valid")
    return value


def _proxy_list(trusted_proxies: Sequence[str]) -> str:
    """Validate IPs/CIDRs and join them for TRUSTED_PROXY_IPS (no catch-all networks)."""
    networks = []
    for item in (p.strip() for raw in trusted_proxies for p in raw.split(",")):
        if not item:
            continue
        try:
            network = ipaddress.ip_network(item, strict=False)
        except ValueError:
            raise ConfigError(f"--trusted-proxy is not an IP or CIDR: {item}") from None
        if network.prefixlen == 0:
            raise ConfigError("--trusted-proxy must not trust every address")
        networks.append(str(network))
    return ",".join(networks)


def space_settings(dotenv: Mapping[str, str], trusted_proxies: Sequence[str]) -> SpaceSettings:
    """Variables and secrets for the Space, from fixed values and `.env` only."""
    provider = dotenv.get("LLM_PROVIDER", "").strip() or "groq"
    if provider not in LLM_PROVIDERS:
        raise ConfigError("LLM_PROVIDER in .env must be groq or anthropic")
    variables = dict(FIXED_VARIABLES)
    variables["LLM_PROVIDER"] = provider
    collection = dotenv.get("QDRANT_COLLECTION", "").strip() or DEFAULT_COLLECTION
    variables["QDRANT_COLLECTION"] = _validated("QDRANT_COLLECTION", collection, COLLECTION_PATTERN)
    repo = dotenv.get("GITHUB_REPO", "").strip()
    if repo:
        variables["GITHUB_REPO"] = _validated("GITHUB_REPO", repo, GITHUB_REPO_PATTERN)
    proxies = _proxy_list(trusted_proxies)
    if proxies:
        variables["TRUSTED_PROXY_IPS"] = proxies

    names = list(SPACE_SECRET_NAMES)
    if provider == "anthropic":
        names.append(ANTHROPIC_SECRET_NAME)
    secrets = {name: dotenv[name].strip() for name in names if is_set(dotenv.get(name))}
    if TOKEN_NAME in secrets or TOKEN_NAME in variables:  # defensive: never upload it
        raise ConfigError("HF_TOKEN must never be a Space setting")
    return SpaceSettings(variables=variables, secrets=secrets)


def describe_plan(
    space: str, files: Sequence[str], settings: SpaceSettings | None
) -> Iterator[str]:
    """Human-readable plan: target, files, and setting *names* only."""
    yield f"Space: {space} (public, Docker SDK)"
    yield f"Files to upload ({len(files)}), mirroring the Space (other files are removed):"
    yield from (f"  {name}" for name in files)
    if settings is None:
        yield "Space settings: unchanged (pass --sync-settings to set them)"
        return
    yield "Space variables: " + ", ".join(sorted(settings.variables))
    yield "Space secrets (values from .env, not shown): " + (
        ", ".join(sorted(settings.secrets)) or "none"
    )
    missing = [n for n in SPACE_SECRET_NAMES if n not in settings.secrets]
    if missing:
        yield "Not set in .env (skipped): " + ", ".join(missing)


def deploy(api: HubApi, space: str, staging: Path, settings: SpaceSettings | None) -> None:
    """Create the Space if needed, mirror the staging folder, then apply the settings."""
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=staging,
        commit_message="Deploy StudioDesk",
        delete_patterns=["*"],  # mirror: drop files no longer staged (.gitattributes kept)
    )
    if settings is None:
        return
    for key, value in sorted(settings.variables.items()):
        api.add_space_variable(space, key, value)
    for key, value in sorted(settings.secrets.items()):
        api.add_space_secret(space, key, value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--space", help="target Space as owner/name (default <you>/studiodesk)")
    parser.add_argument(
        "--dry-run", action="store_true", help="list the files and README; no network"
    )
    parser.add_argument(
        "--sync-settings", action="store_true", help="also set Space variables and secrets"
    )
    parser.add_argument(
        "--trusted-proxy",
        action="append",
        default=[],
        metavar="CIDR",
        help="proxy IP/CIDR for TRUSTED_PROXY_IPS (repeatable; with --sync-settings)",
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument(
        "--env-file", type=Path, default=DEFAULT_ENV_FILE, help="local .env to read"
    )
    return parser.parse_args(argv)


def _default_api(token: str) -> HubApi:
    """A real `HfApi` authenticated with `token` (imported lazily: dry runs need no hub)."""
    from huggingface_hub import HfApi

    api: HubApi = HfApi(token=token)
    return api


def run(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str],
    repo_root: Path = REPO_ROOT,
    api_factory: Callable[[str], HubApi] = _default_api,
    ask: Callable[[str], str] = input,
) -> int:
    """Execute the deploy described by `args`; returns the exit code."""
    dotenv = read_dotenv(args.env_file)
    readme = space_readme(resolve_source_repo_url(dotenv))
    settings = space_settings(dotenv, args.trusted_proxy) if args.sync_settings else None
    if args.trusted_proxy and not args.sync_settings:
        raise ConfigError("--trusted-proxy needs --sync-settings")
    space = validate_space_id(args.space) if args.space else None

    with tempfile.TemporaryDirectory(prefix="studiodesk-space-") as tmp:
        staging = Path(tmp)
        build_staging(repo_root, staging, readme)
        check_staging(staging, secret_values(dotenv, environ))
        files = staged_files(staging)

        if args.dry_run:
            target = space or f"<you>/{DEFAULT_SPACE_NAME}"
            for line in describe_plan(target, files, settings):
                print(line)
            print("\n--- README.md ---\n" + readme)
            return 0

        token = resolve_token(environ, dotenv)
        if token is None:
            raise ConfigError("HF_TOKEN is not set (environment or .env)")
        api = api_factory(token)
        if space is None:
            space = validate_space_id(f"{api.whoami()['name']}/{DEFAULT_SPACE_NAME}")
        for line in describe_plan(space, files, settings):
            print(line)
        if not args.yes and ask("Type yes to deploy: ").strip().lower() != "yes":
            print("aborted: nothing was uploaded")
            return EXIT_ABORTED
        deploy(api, space, staging, settings)

    print(f"Space: {space_url(space)}")
    print(f"App:   {app_url(space)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Errors print the class and a fixed message, never a token."""
    args = parse_args(argv)
    try:
        return run(args, environ=os.environ)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # hub/network errors: report the type only
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f" (HTTP {status})" if status is not None else ""
        print(f"error: Hugging Face request failed: {type(exc).__name__}{suffix}", file=sys.stderr)
        return EXIT_API_ERROR


if __name__ == "__main__":
    sys.exit(main())
