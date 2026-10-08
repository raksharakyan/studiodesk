"""Offline test doubles for the LLM and the GitHub API (no network, no keys)."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, SecretStr

from studiodesk.actions.github import GitHubIssues

Responder = Callable[[str, str], BaseModel]
FAKE_REPO = "studio/sandbox"


@dataclass
class LLMCall:
    """One recorded `structured` call."""

    system: str
    user_content: str
    schema: type[BaseModel]


class FakeLLM:
    """`LLMClient` that answers from per-schema responders and records every call.

    A responder is either a fixed model instance or a callable `(system, user) -> model`;
    an `Exception` instance is raised instead of answering.
    """

    def __init__(self, responses: dict[type[BaseModel], BaseModel | Responder | Exception]):
        self._responses = responses
        self.calls: list[LLMCall] = []

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        self.calls.append(LLMCall(system, user_content, schema))
        response = self._responses[schema]
        if isinstance(response, Exception):
            raise response
        result = response(system, user_content) if callable(response) else response
        assert isinstance(result, schema)
        return result

    def close(self) -> None:
        """Nothing to release."""


@dataclass
class FakeGitHubAPI:
    """MockTransport handler that records requests and answers like `POST .../issues`."""

    repo: str = FAKE_REPO
    status_code: int = 201
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        number = len(self.requests)
        return httpx.Response(
            self.status_code,
            json={"number": number, "html_url": f"https://github.com/{self.repo}/issues/{number}"},
        )

    def payloads(self) -> list[dict[str, object]]:
        """JSON bodies of the recorded requests."""
        return [json.loads(request.content) for request in self.requests]

    def client(self) -> GitHubIssues:
        """A `GitHubIssues` whose HTTP client is routed to this fake."""
        http = httpx.Client(transport=httpx.MockTransport(self))
        return GitHubIssues(http, self.repo, SecretStr("test-token-not-real"), timeout_s=5)
