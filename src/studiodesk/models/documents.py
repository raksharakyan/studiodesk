"""Pydantic models for the documents StudioDesk ingests: bug reports, crash logs, patch notes.

All models forbid unknown fields and bound string lengths, because dataset records and
user submissions are untrusted input.
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SEMVER_PATTERN = r"^\d+\.\d+\.\d+$"
BUG_ID_PATTERN = r"^BUG-\d{4}$"
CRASH_ID_PATTERN = r"^CRASH-\d{4}$"
PATCH_ID_PATTERN = r"^PATCH-\d+\.\d+\.\d+$"

Version = Annotated[str, Field(pattern=SEMVER_PATTERN, max_length=32)]
BugId = Annotated[str, Field(pattern=BUG_ID_PATTERN)]
Title = Annotated[str, Field(min_length=5, max_length=200)]
LongText = Annotated[str, Field(min_length=1, max_length=4_000)]
ShortText = Annotated[str, Field(min_length=1, max_length=500)]


class Platform(StrEnum):
    """Platforms Starfall Outpost ships on."""

    PC = "pc"
    PS5 = "ps5"
    XBOX_SERIES = "xbox_series"
    SWITCH = "switch"


class Severity(StrEnum):
    """Triage severity, most to least severe."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Component(StrEnum):
    """Game subsystem a document relates to."""

    RENDERING = "rendering"
    NETCODE = "netcode"
    SAVE_SYSTEM = "save_system"
    AUDIO = "audio"
    UI = "ui"
    PHYSICS = "physics"
    MATCHMAKING = "matchmaking"
    INPUT = "input"
    PROGRESSION = "progression"
    PERFORMANCE = "performance"


class DocType(StrEnum):
    """Discriminator for document kinds."""

    BUG_REPORT = "bug_report"
    CRASH_LOG = "crash_log"
    PATCH_NOTE = "patch_note"


class _Document(BaseModel):
    """Shared model configuration for all documents."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class BugReport(_Document):
    """A bug report submitted by a player or QA tester."""

    id: BugId
    type: Literal[DocType.BUG_REPORT] = DocType.BUG_REPORT
    title: Title
    description: LongText
    steps_to_reproduce: list[ShortText] = Field(min_length=1, max_length=20)
    expected: ShortText
    actual: ShortText
    platform: Platform
    version: Version
    severity: Severity
    component: Component
    reporter_role: Literal["player", "qa"]
    created_at: datetime
    duplicate_of: BugId | None = None


class CrashLog(_Document):
    """An automatically captured crash report with a stack trace."""

    id: str = Field(pattern=CRASH_ID_PATTERN)
    type: Literal[DocType.CRASH_LOG] = DocType.CRASH_LOG
    platform: Platform
    version: Version
    severity: Severity
    component: Component
    build_hash: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    exception: str = Field(min_length=1, max_length=300)
    stack_trace: str = Field(min_length=1, max_length=8_000)
    summary: LongText
    created_at: datetime
    related_bug: BugId | None = None


class PatchFix(_Document):
    """One fixed bug listed in a patch note."""

    bug_id: BugId
    description: ShortText


class PatchNote(_Document):
    """Release notes for one game version."""

    id: str = Field(pattern=PATCH_ID_PATTERN)
    type: Literal[DocType.PATCH_NOTE] = DocType.PATCH_NOTE
    version: Version
    released_at: date
    summary: LongText
    fixes: list[PatchFix] = Field(max_length=50)
    known_issues: list[ShortText] = Field(default_factory=list, max_length=20)
    platforms: list[Platform] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _id_matches_version(self) -> "PatchNote":
        """Ensure the id is `PATCH-<version>`."""
        if self.id != f"PATCH-{self.version}":
            raise ValueError(f"patch note id {self.id!r} does not match version {self.version!r}")
        return self


def parse_version(version: str) -> tuple[int, int, int]:
    """Convert a `MAJOR.MINOR.PATCH` string into a comparable tuple."""
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch
