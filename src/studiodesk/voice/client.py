"""ElevenLabs Agents API client on an injected `httpx.Client`.

Requests only ever go to the fixed host `https://api.elevenlabs.io`, with the API key in the
`xi-api-key` header (read only when a request is built). Redirects are never followed and
every request has a timeout. Ids placed in URL paths are validated first.

Errors are `ElevenLabsError`: `str(exc)` holds only the exception class or HTTP status, so it
is safe to log. `exc.detail` holds ElevenLabs' own (sanitised, truncated) error message for
the setup CLI; it is never returned by the API. Neither ever contains the key or a signed
URL: a signed URL is a bearer credential for one conversation and must not be logged.
"""

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

ELEVENLABS_API_BASE = "https://api.elevenlabs.io"
ELEVENLABS_HOST = "api.elevenlabs.io"
SIGNED_URL_SCHEME = "wss"
SIGNED_URL_PATH_PREFIX = "/v1/convai/"
# ElevenLabs: a signed URL must be used to start a session within 15 minutes.
SIGNED_URL_TTL_S = 900
SIGNED_URL_MAX_CHARS = 4096
DETAIL_MAX_CHARS = 300
MAX_LIST_PAGES = 20
USER_AGENT = "studiodesk"
# Agent and tool ids are URL-safe tokens; anything else is rejected before a request.
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CONTROL_OR_SPACE = re.compile(r"[\x00-\x1f\x7f-\x9f\s]+")


class ElevenLabsError(RuntimeError):
    """An ElevenLabs call failed. `str(exc)` is safe to log; `detail` is for the CLI only."""

    def __init__(self, message: str, *, status: int | None = None, detail: str | None = None):
        """Store the safe `message`, the HTTP `status` and ElevenLabs' error `detail`."""
        super().__init__(message)
        self.status = status
        self.detail = detail


class RemoteTool(BaseModel):
    """A tool that exists in the ElevenLabs workspace (id and name only)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN.pattern)
    name: str = Field(min_length=1, max_length=128)


def build_http_client(timeout_s: float) -> httpx.Client:
    """Create the outbound client for ElevenLabs.

    No redirects, and `trust_env=False` so HTTP(S)_PROXY, NETRC and other environment
    settings cannot reroute requests or add credentials to them.
    """
    return httpx.Client(timeout=timeout_s, follow_redirects=False, trust_env=False)


def require_id(value: object, what: str) -> str:
    """Return `value` if it is a URL-safe id, else raise `ElevenLabsError`."""
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ElevenLabsError(f"invalid {what}")
    return value


def validate_signed_url(value: object) -> str:
    """Return `value` if it is a `wss://api.elevenlabs.io/v1/convai/...` URL.

    Exact host (no userinfo or port), no whitespace or control characters, no fragment.
    The error message never includes the value.
    """
    if (
        not isinstance(value, str)
        or not value
        or len(value) > SIGNED_URL_MAX_CHARS
        or _CONTROL_OR_SPACE.search(value)
    ):
        raise ElevenLabsError("elevenlabs returned an invalid signed url")
    parts = urlsplit(value)
    if (
        parts.scheme != SIGNED_URL_SCHEME
        or parts.netloc != ELEVENLABS_HOST
        or parts.hostname != ELEVENLABS_HOST
        or parts.fragment
        or "#" in value
        or not parts.path.startswith(SIGNED_URL_PATH_PREFIX)
    ):
        raise ElevenLabsError("elevenlabs returned an invalid signed url")
    return value


def error_detail(response: httpx.Response, secret: str = "") -> str | None:
    """Extract ElevenLabs' error message from a response body, sanitised for printing.

    Handles `{"detail": "..."}`, `{"detail": {"message": ...}}` and validation lists
    `{"detail": [{"msg": ...}]}`. Control characters are collapsed, `secret` is removed and
    the result is truncated. Headers are never read.
    """
    try:
        data = response.json()
    except ValueError:
        return None
    detail = data.get("detail") if isinstance(data, dict) else None
    text = _detail_text(detail)
    if not text:
        return None
    if secret:
        text = text.replace(secret, "[redacted]")
    text = _CONTROL_OR_SPACE.sub(" ", text).strip()
    return text[:DETAIL_MAX_CHARS] or None


def _detail_text(detail: object) -> str | None:
    """Flatten the known shapes of an ElevenLabs/FastAPI `detail` field to text."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("status")
        return message if isinstance(message, str) else None
    if isinstance(detail, list):
        messages = [item.get("msg") for item in detail[:3] if isinstance(item, dict)]
        return "; ".join(m for m in messages if isinstance(m, str)) or None
    return None


class ElevenLabsClient:
    """Calls the ElevenLabs Agents API. The caller owns the HTTP client."""

    def __init__(self, http: httpx.Client, api_key: SecretStr, timeout_s: float) -> None:
        """Use `http` for requests authenticated with `api_key`, each bounded by `timeout_s`."""
        self._http = http
        self._api_key = api_key
        self._timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        """Auth headers (the key is read only here)."""
        return {
            "xi-api-key": self._api_key.get_secret_value(),
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        """Send one request to the pinned host; return the 2xx response or raise."""
        try:
            response = self._http.request(
                method,
                f"{ELEVENLABS_API_BASE}{path}",
                params=params,
                json=json,
                headers=self._headers(),
                timeout=self._timeout_s,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise ElevenLabsError(f"elevenlabs request failed: {type(exc).__name__}") from None
        if not response.is_success:
            raise ElevenLabsError(
                f"elevenlabs returned status {response.status_code}",
                status=response.status_code,
                detail=error_detail(response, self._api_key.get_secret_value()),
            )
        return response

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        """Parse a JSON object body or raise `ElevenLabsError`."""
        try:
            data = response.json()
        except ValueError:
            raise ElevenLabsError("elevenlabs returned invalid JSON") from None
        if not isinstance(data, dict):
            raise ElevenLabsError("elevenlabs returned an unexpected payload")
        return data

    def get_signed_url(self, agent_id: str) -> str:
        """Return a signed `wss://` URL that starts one conversation with `agent_id`.

        The URL is a short-lived bearer credential: never log it.

        Raises:
            ElevenLabsError: on transport errors, a non-2xx status or an invalid URL.
        """
        require_id(agent_id, "agent id")
        response = self._request(
            "GET",
            "/v1/convai/conversation/get-signed-url",
            params={"agent_id": agent_id},
        )
        return validate_signed_url(self._json_object(response).get("signed_url"))

    def list_tools(self) -> list[RemoteTool]:
        """Return every tool in the workspace (follows `next_cursor` while `has_more`)."""
        tools: list[RemoteTool] = []
        params: dict[str, str] = {}
        for _ in range(MAX_LIST_PAGES):
            data = self._json_object(self._request("GET", "/v1/convai/tools", params=params))
            tools.extend(_parse_tools(data.get("tools")))
            cursor = data.get("next_cursor")
            if not data.get("has_more") or not isinstance(cursor, str) or not cursor:
                return tools
            params = {"cursor": cursor}
        raise ElevenLabsError("elevenlabs tool list did not end")

    def create_tool(self, tool_config: Mapping[str, Any]) -> str:
        """Create a tool from `tool_config` and return its id."""
        response = self._request("POST", "/v1/convai/tools", json={"tool_config": tool_config})
        return require_id(self._json_object(response).get("id"), "tool id in response")

    def update_tool(self, tool_id: str, tool_config: Mapping[str, Any]) -> None:
        """Replace the config of the existing tool `tool_id`."""
        path = f"/v1/convai/tools/{require_id(tool_id, 'tool id')}"
        self._request("PATCH", path, json={"tool_config": tool_config})

    def create_agent(self, payload: Mapping[str, Any]) -> str:
        """Create an agent from `payload` and return its `agent_id`."""
        response = self._request("POST", "/v1/convai/agents/create", json=payload)
        return require_id(self._json_object(response).get("agent_id"), "agent id in response")

    def update_agent(self, agent_id: str, payload: Mapping[str, Any]) -> None:
        """Update the existing agent `agent_id` with `payload`."""
        path = f"/v1/convai/agents/{require_id(agent_id, 'agent id')}"
        self._request("PATCH", path, json=payload)


def _parse_tools(items: object) -> list[RemoteTool]:
    """Read `(id, tool_config.name)` pairs from a tool list; skip malformed entries."""
    if not isinstance(items, list):
        raise ElevenLabsError("elevenlabs returned an unexpected tool list")
    tools: list[RemoteTool] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        config = item.get("tool_config")
        name = config.get("name") if isinstance(config, dict) else item.get("name")
        tool_id = item.get("id")
        if (
            isinstance(name, str)
            and 0 < len(name) <= 128
            and isinstance(tool_id, str)
            and ID_PATTERN.fullmatch(tool_id)
        ):
            tools.append(RemoteTool(id=tool_id, name=name))
    return tools
