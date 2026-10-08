"""Create StudioDesk's label allowlist in the GitHub repository from Settings (idempotent).

Reads GITHUB_TOKEN and GITHUB_REPO from the environment / `.env`. Lists the repository's
existing labels (GET, paginated) and creates only the missing allowlisted ones (POST).
Existing labels are never modified or deleted. Requests go only to api.github.com.

Run once by the owner: `uv run python scripts/setup_issue_labels.py`. Not run in tests.
"""

import sys
from collections.abc import Sequence

import httpx
from pydantic import SecretStr, ValidationError

from studiodesk.actions.github import GITHUB_API_BASE, GITHUB_API_VERSION, USER_AGENT
from studiodesk.actions.labels import LABEL_SPECS, LabelSpec
from studiodesk.config import Settings

PER_PAGE = 100
MAX_PAGES = 20


class LabelSyncError(RuntimeError):
    """A GitHub call failed; the message holds only the status or exception class."""


def _headers(token: SecretStr) -> dict[str, str]:
    """GitHub REST headers with the bearer token."""
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token.get_secret_value()}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": USER_AGENT,
    }


def existing_labels(http: httpx.Client, repo: str, token: SecretStr) -> set[str]:
    """Return the names of every label in `repo` (follows page numbers until empty)."""
    names: set[str] = set()
    for page in range(1, MAX_PAGES + 1):
        try:
            response = http.get(
                f"{GITHUB_API_BASE}/repos/{repo}/labels",
                params={"per_page": PER_PAGE, "page": page},
                headers=_headers(token),
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise LabelSyncError(f"listing labels failed: {type(exc).__name__}") from None
        if response.status_code != httpx.codes.OK:
            raise LabelSyncError(f"listing labels returned {response.status_code}")
        batch = [item["name"] for item in response.json() if isinstance(item, dict)]
        names.update(name for name in batch if isinstance(name, str))
        if len(batch) < PER_PAGE:
            break
    return names


def create_label(http: httpx.Client, repo: str, token: SecretStr, spec: LabelSpec) -> None:
    """Create one label; a 422 "already_exists" race counts as success."""
    try:
        response = http.post(
            f"{GITHUB_API_BASE}/repos/{repo}/labels",
            json={"name": spec.name, "color": spec.color, "description": spec.description},
            headers=_headers(token),
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        raise LabelSyncError(f"creating {spec.name} failed: {type(exc).__name__}") from None
    if response.status_code == httpx.codes.CREATED:
        return
    if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY and "already_exists" in (
        response.text
    ):
        return
    raise LabelSyncError(f"creating {spec.name} returned {response.status_code}")


def sync_labels(
    http: httpx.Client, repo: str, token: SecretStr, specs: Sequence[LabelSpec] = LABEL_SPECS
) -> tuple[list[str], list[str]]:
    """Create every missing label in `specs`. Returns `(created, already_present)` names."""
    present = existing_labels(http, repo, token)
    created: list[str] = []
    kept: list[str] = []
    for spec in specs:
        if spec.name in present:
            kept.append(spec.name)
        else:
            create_label(http, repo, token, spec)
            created.append(spec.name)
    return created, kept


def main() -> int:
    """Sync labels in the configured repo and print what changed. Returns the exit code."""
    try:
        settings = Settings()
    except ValidationError as exc:
        for error in exc.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in error["loc"])
            print(f"error: invalid setting {field}: {error['msg']}", file=sys.stderr)
        return 2
    if settings.github_token is None or not settings.github_repo:
        print("error: set GITHUB_TOKEN and GITHUB_REPO (see .env.example)", file=sys.stderr)
        return 2
    with httpx.Client(timeout=settings.github_timeout_s, follow_redirects=False) as http:
        try:
            created, kept = sync_labels(http, settings.github_repo, settings.github_token)
        except LabelSyncError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    print(f"Repository: {settings.github_repo}")
    print(f"Created ({len(created)}): {', '.join(created) or '-'}")
    print(f"Already present ({len(kept)}): {', '.join(kept) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
