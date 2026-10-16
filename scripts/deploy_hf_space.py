"""Deploy StudioDesk to a public Hugging Face Docker Space (idempotent).

Builds a staging directory in a temporary folder from an explicit allowlist (Dockerfile,
.dockerignore, pyproject.toml, uv.lock, the git-tracked files under src/ and data/, plus a
generated Space README with the YAML front matter HF needs). It aborts if src/ or data/
hold untracked or ignored files (other than caches), or if any staged file is named
`.env*`, is a symlink, contains a token-shaped string (GitHub, Groq, Anthropic, HF,
ElevenLabs, Slack) or a secret value from `.env` in raw, base64, URL- or JSON-encoded
form. Then it mirrors the staging folder to the Space with `upload_folder`.

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
import base64
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
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
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_VALUE_RE = re.compile(r"""^(['"])(.*)\1(?:\s+#.*)?$""")
_INLINE_COMMENT_RE = re.compile(r"\s+#")
# Well-known credential shapes; any match in a staged file aborts the deploy.
TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("GitHub fine-grained token", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{22,}")),
    ("Groq key", re.compile(rb"\bgsk_[A-Za-z0-9]{20,}")),
    ("Anthropic key", re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{30,}")),
    ("ElevenLabs key", re.compile(rb"\bsk_[0-9a-f]{32,}")),
    ("Slack token", re.compile(rb"\bxox[abp]-[A-Za-z0-9-]{10,}")),
    ("Slack webhook", re.compile(rb"hooks\.slack\.com/services/")),
)

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
    """Parse a `.env` file the way pydantic-settings does (python-dotenv), else locally.

    No `${VAR}` interpolation (the shell environment must not leak in). A missing file
    yields an empty dict. Values are returned, never printed.
    """
    if not path.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv ships with pydantic-settings
        return _parse_dotenv(path)
    parsed = dotenv_values(path, interpolate=False, encoding="utf-8")
    return {k: v or "" for k, v in parsed.items() if k and _ENV_NAME_RE.fullmatch(k)}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Fallback `.env` parser: KEY=VALUE, `export`, quotes and inline `# comments`."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _DOTENV_LINE_RE.match(line)
        if match is None:
            continue
        key, value = match.groups()
        quoted = _QUOTED_VALUE_RE.match(value)
        if quoted is not None:
            value = quoted.group(2)
        else:
            value = _INLINE_COMMENT_RE.split(value, maxsplit=1)[0].rstrip()
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


def _is_cache_path(relative: str) -> bool:
    """True for caches, bytecode and OS litter (never staged, never reported)."""
    return any(_skipped(Path(part)) for part in Path(relative).parts)


def _git_ls_files(repo_root: Path, *options: str) -> list[str]:
    """`git ls-files -z <options> -- src data` in `repo_root`, as relative POSIX paths."""
    git = shutil.which("git")
    if git is None:
        raise ConfigError("git is required: only files tracked by git are staged")
    try:
        result = subprocess.run(  # noqa: S603 - fixed binary and arguments
            [git, "-C", str(repo_root), "ls-files", "-z", *options, "--", *STAGED_DIRS],
            capture_output=True,
            check=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise ConfigError("git ls-files failed: run the deploy from a git checkout") from None
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def tracked_files(repo_root: Path) -> list[str]:
    """Git-tracked files under src/ and data/ (caches excluded).

    Raises:
        ConfigError: listing any untracked or ignored non-cache files under src/ or data/
            (commit or delete them first), so nothing unreviewed is ever uploaded.
    """
    stray = [p for p in _git_ls_files(repo_root, "--others") if not _is_cache_path(p)]
    if stray:
        raise ConfigError(
            "untracked or ignored files under src/ or data/ (commit or remove them): "
            + ", ".join(sorted(stray))
        )
    return sorted(p for p in _git_ls_files(repo_root) if not _is_cache_path(p))


def _copy_file(repo_root: Path, relative: str, staging: Path) -> None:
    """Copy one regular file into `staging`, refusing symlinks and missing files."""
    source = repo_root / relative
    if source.is_symlink():
        raise ConfigError(f"refusing to stage a symlink: {relative}")
    if not source.is_file():
        raise ConfigError(f"missing, deleted or not a regular file: {relative}")
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build_staging(repo_root: Path, staging: Path, readme: str) -> None:
    """Fill `staging` with the allowlisted files, git-tracked src/ and data/, and README."""
    for name in STAGED_FILES:
        _copy_file(repo_root, name, staging)
    for relative in tracked_files(repo_root):
        _copy_file(repo_root, relative, staging)
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


def encoded_forms(secret: str) -> set[bytes]:
    """The secret raw and as base64 (standard and URL-safe, unpadded), URL-encoded and
    JSON-escaped, so a staged copy in any of those forms is caught too."""
    raw = secret.encode("utf-8")
    return {
        raw,
        base64.b64encode(raw).rstrip(b"="),
        base64.urlsafe_b64encode(raw).rstrip(b"="),
        urllib.parse.quote(secret, safe="").encode("ascii"),
        json.dumps(secret)[1:-1].encode("utf-8"),
    }


def scan_bytes(data: bytes, needles: Sequence[bytes]) -> str | None:
    """Why `data` must not be uploaded (a known token shape or a secret value), or None.

    The reason names the kind of match only, never the matched text.
    """
    for name, pattern in TOKEN_PATTERNS:
        if pattern.search(data):
            return f"looks like a {name}"
    if any(needle in data for needle in needles):
        return "contains a secret value from .env (raw or encoded)"
    return None


def check_staging(staging: Path, secrets: Sequence[str]) -> None:
    """Fail closed if any staged path is `.env*`, a symlink, contains a token-shaped
    string, or contains a secret value in raw, base64, URL- or JSON-encoded form.

    Error messages name the file and the kind of match, never the secret.
    """
    needles = sorted({form for secret in secrets for form in encoded_forms(secret)})
    for path in sorted(staging.rglob("*")):
        relative = path.relative_to(staging).as_posix()
        if path.is_symlink():
            raise ConfigError(f"symlink in staging: {relative}")
        if path.name.startswith(FORBIDDEN_NAME_PREFIX):
            raise ConfigError(f".env file in staging: {relative}")
        if path.is_file():
            reason = scan_bytes(path.read_bytes(), needles)
            if reason is not None:
                raise ConfigError(f"{relative} {reason}")


def _validated(name: str, value: str, pattern: str) -> str:
    """Return `value` if it matches `pattern`, else raise ConfigError naming the setting."""
    if not re.fullmatch(pattern, value):
        raise ConfigError(f"{name} in .env is not valid")
    return value


# Address ranges a platform's internal proxy can legitimately use. A public network as a
# trusted proxy would let anyone in it set X-Forwarded-For (needs --force-public-proxy).
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "fc00::/7",
        "::1/128",
    )
)


def is_private_network(network: ipaddress.IPv4Network | ipaddress.IPv6Network) -> bool:
    """True if `network` lies entirely inside one of PRIVATE_NETWORKS."""
    return any(
        network.version == private.version and network.subnet_of(private)  # type: ignore[arg-type]
        for private in PRIVATE_NETWORKS
    )


def _proxy_list(trusted_proxies: Sequence[str], *, allow_public: bool = False) -> str:
    """Validate IPs/CIDRs and join them for TRUSTED_PROXY_IPS.

    Catch-all networks are always rejected; public (non-private) ones unless
    `allow_public`.
    """
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
        if not allow_public and not is_private_network(network):
            raise ConfigError(
                f"--trusted-proxy {network} is not a private network "
                "(pass --force-public-proxy if you are sure)"
            )
        networks.append(str(network))
    return ",".join(networks)


def space_settings(
    dotenv: Mapping[str, str],
    trusted_proxies: Sequence[str],
    *,
    allow_public_proxy: bool = False,
    debug_client_ip: bool = False,
) -> SpaceSettings:
    """Variables and secrets for the Space, from fixed values and `.env` only."""
    provider = dotenv.get("LLM_PROVIDER", "").strip() or "groq"
    if provider not in LLM_PROVIDERS:
        raise ConfigError("LLM_PROVIDER in .env must be groq or anthropic")
    variables = dict(FIXED_VARIABLES)
    variables["LLM_PROVIDER"] = provider
    # Always written, so a later sync without --debug-client-ip turns the log off again.
    variables["DEBUG_LOG_CLIENT_IP"] = "true" if debug_client_ip else "false"
    collection = dotenv.get("QDRANT_COLLECTION", "").strip() or DEFAULT_COLLECTION
    variables["QDRANT_COLLECTION"] = _validated("QDRANT_COLLECTION", collection, COLLECTION_PATTERN)
    repo = dotenv.get("GITHUB_REPO", "").strip()
    if repo:
        variables["GITHUB_REPO"] = _validated("GITHUB_REPO", repo, GITHUB_REPO_PATTERN)
    proxies = _proxy_list(trusted_proxies, allow_public=allow_public_proxy)
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
    parser.add_argument(
        "--force-public-proxy",
        action="store_true",
        help="allow a non-private network in --trusted-proxy",
    )
    parser.add_argument(
        "--debug-client-ip",
        action="store_true",
        help="set DEBUG_LOG_CLIENT_IP=true (log peer IP + X-Forwarded-For to find proxy hops)",
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
    if (args.trusted_proxy or args.debug_client_ip) and not args.sync_settings:
        raise ConfigError("--trusted-proxy and --debug-client-ip need --sync-settings")
    settings = (
        space_settings(
            dotenv,
            args.trusted_proxy,
            allow_public_proxy=args.force_public_proxy,
            debug_client_ip=args.debug_client_ip,
        )
        if args.sync_settings
        else None
    )
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
