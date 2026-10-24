"""Scan bytes for credentials before anything is published (Space upload, static site).

Shared by `scripts/deploy_hf_space.py`, `scripts/record_showcase.py` and
`scripts/build_site.py`. A scan fails on any well-known token shape (GitHub, Groq,
Anthropic, Hugging Face, ElevenLabs, Slack) and on any configured secret value in raw,
base64, URL- or JSON-encoded form. Reasons name the kind of match, never the matched text.
"""

import base64
import json
import re
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from studiodesk.config import PLACEHOLDER_VALUES

# Env var names whose values count as secrets.
SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|WEBHOOK)", re.IGNORECASE)
# Shorter values would match ordinary text by accident.
MIN_SCANNED_SECRET_CHARS = 8

_DOTENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_VALUE_RE = re.compile(r"""^(['"])(.*)\1(?:\s+#.*)?$""")
_INLINE_COMMENT_RE = re.compile(r"\s+#")

# Well-known credential shapes; any match fails the scan.
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


def scannable(values: Iterable[str | None]) -> list[str]:
    """The set, non-placeholder values long enough to scan for, stripped and sorted."""
    stripped = {value.strip() for value in values if value is not None}
    return sorted(v for v in stripped if is_set(v) and len(v) >= MIN_SCANNED_SECRET_CHARS)


def secret_like_values(env: Mapping[str, str], extra_names: Iterable[str] = ()) -> list[str]:
    """Values of every secret-like entry of `env` plus the `extra_names` entries."""
    names = {k for k in env if SECRET_NAME_RE.search(k)} | set(extra_names)
    return scannable(env.get(name) for name in names)


def encoded_forms(secret: str) -> set[bytes]:
    """The secret raw and as base64 (standard and URL-safe, unpadded), URL-encoded and
    JSON-escaped, so a copy in any of those forms is caught too."""
    raw = secret.encode("utf-8")
    return {
        raw,
        base64.b64encode(raw).rstrip(b"="),
        base64.urlsafe_b64encode(raw).rstrip(b"="),
        urllib.parse.quote(secret, safe="").encode("ascii"),
        json.dumps(secret)[1:-1].encode("utf-8"),
    }


def needles_for(secrets: Iterable[str]) -> list[bytes]:
    """Every encoded form of every secret, deduplicated and sorted."""
    return sorted({form for secret in secrets for form in encoded_forms(secret)})


def scan_bytes(data: bytes, needles: Sequence[bytes]) -> str | None:
    """Why `data` must not be published (a known token shape or a secret value), or None.

    The reason names the kind of match only, never the matched text.
    """
    for name, pattern in TOKEN_PATTERNS:
        if pattern.search(data):
            return f"looks like a {name}"
    if any(needle in data for needle in needles):
        return "contains a secret value from .env (raw or encoded)"
    return None


def scan_json(document: object, needles: Sequence[bytes], path: str = "$") -> str | None:
    """Scan every key and string in a parsed JSON document; return the first finding.

    The finding names the JSON path (e.g. `$.questions[0].response.answer`) and the kind
    of match, never the matched text.
    """
    if isinstance(document, str):
        reason = scan_bytes(document.encode("utf-8"), needles)
        return f"{path} {reason}" if reason is not None else None
    if isinstance(document, dict):
        for key, value in document.items():
            child = f"{path}.{key}"
            finding = scan_json(str(key), needles, f"{child} (key)") or scan_json(
                value, needles, child
            )
            if finding is not None:
                return finding
    elif isinstance(document, list):
        for index, item in enumerate(document):
            finding = scan_json(item, needles, f"{path}[{index}]")
            if finding is not None:
                return finding
    return None
