"""Models for proposed and confirmed outward actions (filing a GitHub issue)."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from studiodesk.actions.labels import LABEL_ALLOWLIST

ISSUE_TITLE_MAX_CHARS = 256
ISSUE_BODY_MAX_CHARS = 12_000
ISSUE_MAX_LABELS = 10
# secrets.token_urlsafe(32) is 43 chars; accept only URL-safe base64 of a sane length.
CONFIRM_TOKEN_PATTERN = r"^[A-Za-z0-9_-]{32,128}$"  # noqa: S105 (a regex, not a secret)
ACTION_ID_PATTERN = r"^[A-Za-z0-9_-]{16,64}$"

ActionId = Annotated[str, StringConstraints(pattern=ACTION_ID_PATTERN)]
ConfirmToken = Annotated[str, StringConstraints(pattern=CONFIRM_TOKEN_PATTERN)]


class IssueDraft(BaseModel):
    """The exact issue that will be created on confirm. Built server-side only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=ISSUE_TITLE_MAX_CHARS)
    body: str = Field(min_length=1, max_length=ISSUE_BODY_MAX_CHARS)
    labels: list[str] = Field(default_factory=list, max_length=ISSUE_MAX_LABELS)

    @field_validator("labels")
    @classmethod
    def _labels_in_allowlist(cls, labels: list[str]) -> list[str]:
        """Reject any label outside the fixed allowlist."""
        unknown = [label for label in labels if label not in LABEL_ALLOWLIST]
        if unknown:
            raise ValueError("labels must come from the label allowlist")
        return labels


class ProposedAction(BaseModel):
    """A pending action the user must confirm with `confirm_token` before `expires_at`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    kind: Literal["file_issue"] = "file_issue"
    confirm_token: str
    expires_at: datetime
    repo: str
    preview: IssueDraft


class ConfirmRequest(BaseModel):
    """Body of `POST /actions/{action_id}/confirm` and `/cancel`: only the token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    confirm_token: ConfirmToken


class SlackStatus(StrEnum):
    """Outcome of the Slack alert sent after an issue is filed."""

    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


class CreatedIssue(BaseModel):
    """An issue GitHub reports as created."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(gt=0)
    url: str


class ConfirmResponse(BaseModel):
    """Body returned by a successful confirm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["confirmed"] = "confirmed"
    issue_number: int
    issue_url: str
    slack: SlackStatus


class CancelResponse(BaseModel):
    """Body returned by a successful cancel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["cancelled"] = "cancelled"
