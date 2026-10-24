"""Build the static GitHub Pages showcase into `_site/` (no network).

Copies `site/` to `_site/` and `src/studiodesk/static/styles.css` to `_site/assets/`, then
fails (and removes `_site/`) unless `_site/data/showcase.json` matches the
`studiodesk.showcase.Showcase` schema and no file contains a token-shaped string or a
secret value from `.env` (raw, base64, URL- or JSON-encoded; skipped when there is no
`.env`, as in CI). Symlinks in `site/` are refused.

    uv run python scripts/build_site.py [--site-dir site] [--out _site]

Exit codes: 0 ok, 2 invalid input (schema, secret scan, layout).
"""

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from studiodesk.secretscan import needles_for, read_dotenv, scan_bytes, scan_json
from studiodesk.secretscan import secret_like_values as secret_like
from studiodesk.showcase import Showcase

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SITE_DIR = REPO_ROOT / "site"
DEFAULT_OUT = REPO_ROOT / "_site"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
STATIC_DIR = REPO_ROOT / "src" / "studiodesk" / "static"
# Files from the app's static folder the site reuses, copied to `_site/assets/`.
SHARED_ASSETS = ("styles.css",)
SHOWCASE_PATH = Path("data") / "showcase.json"
EXTRA_SECRET_NAMES = ("QDRANT_URL", "ELEVENLABS_AGENT_ID", "SLACK_WEBHOOK_URL")

EXIT_INVALID = 2


class BuildError(Exception):
    """The site cannot be built. Messages never contain secret values."""


def check_layout(site_dir: Path, out: Path) -> None:
    """Refuse a missing site dir, symlinks in it, or an output that overlaps the input."""
    if not site_dir.is_dir():
        raise BuildError(f"site directory not found: {site_dir}")
    for path in site_dir.rglob("*"):
        if path.is_symlink():
            raise BuildError(f"symlink in site/: {path.relative_to(site_dir).as_posix()}")
    source, target = site_dir.resolve(), out.resolve()
    if target == source or target in source.parents or source in target.parents:
        raise BuildError("--out must not contain or be inside the site directory")


def copy_site(site_dir: Path, out: Path, static_dir: Path) -> None:
    """Replace `out` with a copy of `site_dir` plus the shared assets."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(site_dir, out)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    for name in SHARED_ASSETS:
        source = static_dir / name
        if not source.is_file():
            raise BuildError(f"shared asset not found: {name}")
        shutil.copyfile(source, assets / name)


def load_showcase(path: Path) -> object:
    """Parse and schema-check the showcase JSON; return the parsed document."""
    if not path.is_file():
        raise BuildError(f"missing {SHOWCASE_PATH.as_posix()}")
    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BuildError(f"{SHOWCASE_PATH.as_posix()} is not valid UTF-8 JSON") from None
    try:
        Showcase.model_validate(document)
    except ValidationError as exc:
        locations = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
        raise BuildError(f"showcase schema check failed at: {', '.join(locations)}") from None
    return document


def scan_site(out: Path, document: object, secrets: Sequence[str]) -> None:
    """Fail on a token shape or secret value in the showcase fields or any built file."""
    needles = needles_for(secrets)
    finding = scan_json(document, needles)
    if finding is not None:
        raise BuildError(f"{SHOWCASE_PATH.as_posix()}: {finding}")
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        reason = scan_bytes(path.read_bytes(), needles)
        if reason is not None:
            raise BuildError(f"{path.relative_to(out).as_posix()} {reason}")


def build(site_dir: Path, out: Path, static_dir: Path, secrets: Sequence[str]) -> int:
    """Build and check the site; on any failure remove `out` and re-raise. Returns files."""
    check_layout(site_dir, out)
    try:
        copy_site(site_dir, out, static_dir)
        document = load_showcase(out / SHOWCASE_PATH)
        scan_site(out, document, secrets)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return sum(1 for p in out.rglob("*") if p.is_file())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR, help="site source")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="build output")
    parser.add_argument(
        "--env-file", type=Path, default=DEFAULT_ENV_FILE, help=".env whose secrets to scan for"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    secrets = secret_like(read_dotenv(args.env_file), EXTRA_SECRET_NAMES)
    try:
        count = build(args.site_dir, args.out, STATIC_DIR, secrets)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    print(f"built {args.out} ({count} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
