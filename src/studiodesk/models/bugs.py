"""Models for `POST /bugs/check`: the submitted report, duplicate verdict and routing.

`DuplicateJudgements` is the LLM's structured output; like `LLMAnswer` it has no length or
range constraints (structured outputs cannot enforce them), so the server clamps
`confidence` and truncates `reason` after parsing.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from studiodesk.models.actions import ProposedAction
from studiodesk.models.documents import (
    Component,
    LongText,
    Platform,
    Severity,
    ShortText,
    Title,
    Version,
)

REASON_MAX_CHARS = 300
MAX_STEPS = 20


class NewBugReport(BaseModel):
    """A bug report submitted for duplicate checking and filing. All fields are untrusted."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    title: Title
    description: LongText
    steps_to_reproduce: list[ShortText] = Field(default_factory=list, max_length=MAX_STEPS)
    expected: ShortText | None = None
    actual: ShortText | None = None
    platform: Platform
    version: Version

    @property
    def search_text(self) -> str:
        """Text embedded for duplicate search and routing: title plus description."""
        return f"{self.title}\n{self.description}"

    def prompt_text(self) -> str:
        """All report fields as plain text, for the (delimited) duplicate prompt."""
        lines = [
            f"Title: {self.title}",
            f"Platform: {self.platform.value}",
            f"Version: {self.version}",
            f"Description: {self.description}",
        ]
        lines += [f"Step {i}: {step}" for i, step in enumerate(self.steps_to_reproduce, 1)]
        if self.expected:
            lines.append(f"Expected: {self.expected}")
        if self.actual:
            lines.append(f"Actual: {self.actual}")
        return "\n".join(lines)


class CandidateJudgement(BaseModel):
    """The LLM's judgement on one candidate (structured output item)."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(description="The candidate document's id attribute.")
    is_duplicate: bool = Field(description="True if it is the same underlying defect.")
    confidence: float = Field(description="Confidence from 0 to 1.")
    reason: str = Field(description="One short sentence.")


class DuplicateJudgements(BaseModel):
    """Structured output requested from the LLM for duplicate judging."""

    model_config = ConfigDict(extra="forbid")

    judgements: list[CandidateJudgement]


class DuplicateVerdict(StrEnum):
    """Advisory duplicate verdict."""

    DUPLICATE = "duplicate"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    NEW = "new"


class DuplicateCandidate(BaseModel):
    """An existing bug report scored against the new one. `title`/`reason` are untrusted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    title: str
    score: float
    is_duplicate: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=REASON_MAX_CHARS)


class DuplicateCheck(BaseModel):
    """Result of duplicate detection (advisory only; nothing is changed by it)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: DuplicateVerdict
    # Canonical original (root of the `duplicate_of` chain) of the matched report.
    duplicate_of: str | None = None
    # The candidate that actually matched (may itself be a labelled duplicate).
    matched_report: str | None = None
    candidates: list[DuplicateCandidate] = Field(default_factory=list)
    # Score from which a candidate counts as flagged even if the LLM said no.
    auto_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    # Candidate doc id -> canonical original id (server-side only, not in API responses).
    canonical_ids: dict[str, str] = Field(default_factory=dict)

    def canonical(self, doc_id: str) -> str:
        """Canonical original of candidate `doc_id` (itself if unknown or not a duplicate)."""
        return self.canonical_ids.get(doc_id, doc_id)


class RoutingResult(BaseModel):
    """Predicted component and severity from a kNN vote over similar bug reports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: Component | None
    component_share: float = Field(ge=0.0, le=1.0)
    severity: Severity | None
    severity_share: float = Field(ge=0.0, le=1.0)
    neighbours: list[str]
    labels: list[str]


class BugCheckResponse(BaseModel):
    """Body returned by `POST /bugs/check`.

    `duplicate_of` is the canonical original (the root of the matched report's
    `duplicate_of` chain); `matched_report` is the report that actually matched.
    `proposed_action` is set unless the verdict is `duplicate`; nothing happens until it is
    confirmed via `POST /actions/{action_id}/confirm`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: DuplicateVerdict
    duplicate_of: str | None
    matched_report: str | None
    candidates: list[DuplicateCandidate]
    routing: RoutingResult
    proposed_action: ProposedAction | None
