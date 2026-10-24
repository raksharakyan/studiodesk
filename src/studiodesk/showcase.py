"""Schema of `site/data/showcase.json`, the recorded results the static site renders.

This is the contract between `scripts/record_showcase.py` (writes it),
`scripts/build_site.py` (validates it) and the static site under `site/` (reads it).
`/ask` and `/bugs/check` responses are validated with the API's own response models, so a
recording always matches what the live API returns. Confirm tokens, action ids and expiry
times of proposed issues must be redacted (null): a recording can never be confirmed.
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from studiodesk.models.answer import AnswerResponse
from studiodesk.models.bugs import BugCheckRequest, BugCheckResponse

SCHEMA_VERSION = 1
BUG_CHECK_IDS = ("duplicate", "new", "injection")
REDACTED_ACTION_FIELDS = ("action_id", "confirm_token", "expires_at")
ISSUE_URL_PATTERN = r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[1-9][0-9]*$"
SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
TEXT_MAX_CHARS = 2_000

Label = Annotated[str, StringConstraints(min_length=1, max_length=120)]
Slug = Annotated[str, StringConstraints(pattern=SLUG_PATTERN)]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    """Base for every showcase model: unknown keys are rejected, instances are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShowcaseModels(_Strict):
    """Which models produced the recording."""

    llm: Label
    embedding: Label


class DatasetCounts(_Strict):
    """Number of documents of each type in the synthetic dataset."""

    bug_reports: int = Field(ge=0)
    crash_logs: int = Field(ge=0)
    patch_notes: int = Field(ge=0)
    docs: int = Field(ge=0)


class RecordedQuestion(_Strict):
    """One main-page question chip and the exact `/ask` response."""

    id: Slug
    label: Label
    question: Annotated[str, StringConstraints(min_length=1, max_length=1_000)]
    response: AnswerResponse


def _require_redacted(response: BugCheckResponse) -> BugCheckResponse:
    """Reject a response whose proposed action still carries an id, token or expiry."""
    action = response.proposed_action
    if action is not None and any(getattr(action, f) is not None for f in REDACTED_ACTION_FIELDS):
        raise ValueError("proposed_action must have action_id, confirm_token, expires_at null")
    return response


class RecordedBugCheck(_Strict):
    """One "Try a ..." bug example and the redacted `/bugs/check` response."""

    id: Literal["duplicate", "new", "injection"]
    label: Label
    request: BugCheckRequest
    response: BugCheckResponse

    @field_validator("response")
    @classmethod
    def _redacted(cls, response: BugCheckResponse) -> BugCheckResponse:
        """Tokens, action ids and expiry times must be null."""
        return _require_redacted(response)


class RecordedFileAnyway(_Strict):
    """The duplicate example re-sent with `file_anyway: true` (redacted like bug checks)."""

    request: BugCheckRequest
    response: BugCheckResponse

    @field_validator("response")
    @classmethod
    def _redacted(cls, response: BugCheckResponse) -> BugCheckResponse:
        """Tokens, action ids and expiry times must be null."""
        return _require_redacted(response)

    @field_validator("request")
    @classmethod
    def _file_anyway_set(cls, request: BugCheckRequest) -> BugCheckRequest:
        """The request must actually opt in to filing a duplicate."""
        if not request.file_anyway:
            raise ValueError("file_anyway request must set file_anyway: true")
        return request


class FiledIssueExample(_Strict):
    """A real issue filed earlier through the confirm step (link only)."""

    url: Annotated[str, StringConstraints(pattern=ISSUE_URL_PATTERN)]
    title: Label


class TranscriptTurn(_Strict):
    """One turn of the recorded voice conversation."""

    role: Literal["agent", "user"]
    text: Annotated[str, StringConstraints(min_length=1, max_length=TEXT_MAX_CHARS)]


class EvalSummary(_Strict):
    """Headline eval numbers (from `evals/reports/latest.md`)."""

    retrieval_hit_at_5: Ratio
    citation_validity: Ratio
    duplicates_heldout_precision: Ratio
    duplicates_heldout_recall: Ratio
    injection_pass: Annotated[str, StringConstraints(pattern=r"^[0-9]{1,4}/[0-9]{1,4}$")]
    routing_component_accuracy: Ratio


class Showcase(_Strict):
    """The whole `showcase.json` document."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal[1] = Field(alias="schema")
    recorded_at: AwareDatetime
    models: ShowcaseModels
    dataset: DatasetCounts
    questions: list[RecordedQuestion] = Field(min_length=1, max_length=20)
    bug_checks: list[RecordedBugCheck] = Field(min_length=3, max_length=3)
    file_anyway: RecordedFileAnyway
    filed_issue_example: FiledIssueExample
    voice_transcript: list[TranscriptTurn] = Field(min_length=1, max_length=50)
    evals: EvalSummary

    @field_validator("recorded_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        """`recorded_at` must be in UTC."""
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("recorded_at must be UTC")
        return value

    @model_validator(mode="after")
    def _unique_ids(self) -> "Showcase":
        """Question ids are unique and the bug checks are exactly the three examples."""
        question_ids = [q.id for q in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question ids must be unique")
        if sorted(b.id for b in self.bug_checks) != sorted(BUG_CHECK_IDS):
            raise ValueError(f"bug_checks must be exactly {', '.join(BUG_CHECK_IDS)}")
        return self


def redact_bug_check(response: dict[str, Any]) -> dict[str, Any]:
    """Copy of a `/bugs/check` JSON body with the proposal's id, token and expiry nulled.

    Everything else (verdict, candidates, routing, the issue preview) is left as returned.
    """
    redacted = dict(response)
    action = redacted.get("proposed_action")
    if isinstance(action, dict):
        redacted["proposed_action"] = {
            **action,
            **dict.fromkeys(REDACTED_ACTION_FIELDS),
        }
    return redacted
