"""Offline test doubles for the LLM and the GitHub API (no network, no keys)."""

import json
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, SecretStr

from studiodesk.actions.github import GitHubIssues
from studiodesk.models.bugs import CandidateJudgement, DuplicateJudgements
from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import Component, DocType, Severity
from studiodesk.vectorstore import ScoredChunk

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


def make_chunk(
    doc_id: str,
    *,
    title: str = "A bug title",
    text: str = "Some bug text.",
    doc_type: DocType = DocType.BUG_REPORT,
    component: Component | None = None,
    severity: Severity | None = None,
) -> Chunk:
    """A minimal valid chunk for unit tests."""
    return Chunk(
        doc_id=doc_id,
        doc_type=doc_type,
        chunk_index=0,
        title=title,
        text=text,
        component=component,
        severity=severity,
    )


def scored(doc_id: str, score: float, **chunk_fields: Any) -> ScoredChunk:
    """A `ScoredChunk` around `make_chunk`."""
    return ScoredChunk(chunk=make_chunk(doc_id, **chunk_fields), score=score)


@dataclass
class FakeRetriever:
    """Retriever double returning fixed hits and recording every search."""

    hits: list[ScoredChunk] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)

    def search(
        self,
        text: str,
        filters: object,
        top_k: int,
        *,
        exclude_ids: Collection[str] = (),
    ) -> list[ScoredChunk]:
        self.searches.append(
            {"text": text, "filters": filters, "top_k": top_k, "exclude": set(exclude_ids)}
        )
        return [h for h in self.hits if h.chunk.doc_id not in exclude_ids][:top_k]

    def search_bug_reports(
        self, text: str, top_k: int, *, exclude_ids: Collection[str] = ()
    ) -> list[ScoredChunk]:
        return self.search(text, "bug_reports_only", top_k, exclude_ids=exclude_ids)


def prompt_doc_ids(user_content: str) -> list[str]:
    """Doc ids of the `<document id="...">` tags in a built prompt, in order."""
    return [part.split('"')[0] for part in user_content.split('<document id="')[1:]]


def judge_all(
    is_duplicate: bool, *, confidence: float = 0.9, reason: str = "MODEL-REASON-TEXT"
) -> Responder:
    """Responder judging every candidate in the prompt the same way."""

    def respond(system: str, user: str) -> DuplicateJudgements:
        return DuplicateJudgements(
            judgements=[
                CandidateJudgement(
                    candidate_id=doc_id,
                    is_duplicate=is_duplicate,
                    confidence=confidence,
                    reason=reason,
                )
                for doc_id in prompt_doc_ids(user)
            ]
        )

    return respond
