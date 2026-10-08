"""Create GitHub issues through the REST API on an injected `httpx.Client`.

Requests only ever go to the fixed host `https://api.github.com`, to
`/repos/{github_repo}/issues` with the repo from Settings. Redirects are not followed and
every request has a timeout. Errors carry only the exception class or HTTP status, never
the token, headers or response body.
"""

import httpx
from pydantic import SecretStr

from studiodesk.models.actions import CreatedIssue, IssueDraft

GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
USER_AGENT = "studiodesk"


class GitHubError(RuntimeError):
    """Issue creation failed. The message is safe to log, not to return."""


class GitHubIssues:
    """Files issues in one repository. The caller owns the HTTP client."""

    def __init__(self, http: httpx.Client, repo: str, token: SecretStr, timeout_s: float) -> None:
        """Target `repo` (`owner/name`, validated by Settings) with `token`.

        Raises:
            ValueError: if `repo` is not exactly `owner/name` with no dot segments.
        """
        parts = repo.split("/")
        if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
            raise ValueError("repo must be owner/name")
        self._http = http
        self._token = token
        self._timeout_s = timeout_s
        self.repo = repo

    @property
    def issues_url(self) -> str:
        """The fixed endpoint issues are created at."""
        return f"{GITHUB_API_BASE}/repos/{self.repo}/issues"

    def _headers(self) -> dict[str, str]:
        """Auth and API-version headers (the token is read only here)."""
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token.get_secret_value()}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": USER_AGENT,
        }

    def create(self, draft: IssueDraft) -> CreatedIssue:
        """Create `draft` as an issue and return its number and html URL.

        Raises:
            GitHubError: on transport errors, a non-201 status or an unexpected response.
        """
        try:
            response = self._http.post(
                self.issues_url,
                json={"title": draft.title, "body": draft.body, "labels": draft.labels},
                headers=self._headers(),
                timeout=self._timeout_s,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise GitHubError(f"github request failed: {type(exc).__name__}") from None
        if response.status_code != httpx.codes.CREATED:
            raise GitHubError(f"github returned status {response.status_code}")
        return self._parse_created(response)

    def _parse_created(self, response: httpx.Response) -> CreatedIssue:
        """Validate the 201 body: an integer number and an html_url inside this repo."""
        try:
            data = response.json()
        except ValueError:
            raise GitHubError("github returned invalid JSON") from None
        number = data.get("number") if isinstance(data, dict) else None
        url = data.get("html_url") if isinstance(data, dict) else None
        expected_prefix = f"https://github.com/{self.repo}/issues/"
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number <= 0
            or not isinstance(url, str)
            or url != f"{expected_prefix}{number}"
        ):
            raise GitHubError("github returned an unexpected issue payload")
        return CreatedIssue(number=number, url=url)
