"""Typed loading and validation of the labelled eval cases in `evals/cases/*.jsonl`.

One JSON object per line, each with `id`, `suite`, `split`, `inputs` and `labels`. Every
model forbids unknown fields, so a typo in a case file fails loudly instead of being
silently ignored. Case text (questions, bug reports) is untrusted data: it is only ever
passed to the agent the same way the API would pass user input.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from studiodesk.models.bugs import NewBugReport
from studiodesk.models.documents import Component, Platform, Severity

CASES_DIR = Path(__file__).resolve().parent / "cases"
CASE_FILES = {
    "retrieval": "retrieval.jsonl",
    "answer": "answer.jsonl",
    "duplicates": "duplicates.jsonl",
    "injection": "injection.jsonl",
}
MAX_KEY_FACTS = 3

Split = Literal["calibration", "heldout"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QuestionInputs(_Strict):
    """A question asked as a player, support agent or developer would ask it."""

    question: str = Field(min_length=1, max_length=1_000)


class ReportInputs(_Strict):
    """A bug report submitted for a duplicate check (mirrors `NewBugReport`)."""

    title: str = Field(min_length=5, max_length=200)
    description: str = Field(min_length=1, max_length=4_000)
    platform: Platform
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    exclude_ids: list[str] = Field(default_factory=list)

    def to_report(self) -> NewBugReport:
        """The `NewBugReport` the API would build from this submission."""
        return NewBugReport(
            title=self.title,
            description=self.description,
            platform=self.platform,
            version=self.version,
        )


class RetrievalLabels(_Strict):
    """Doc ids that contain the answer; any of them in the top k counts as a hit."""

    relevant_ids: list[str] = Field(min_length=1)
    platform: Platform | None = None
    version: str | None = None
    voice: Literal["player", "dev", "support"]


class RetrievalCase(_Strict):
    id: str
    suite: Literal["retrieval"]
    split: Split
    inputs: QuestionInputs
    labels: RetrievalLabels


class AnswerLabels(_Strict):
    """Expected sources and key facts; each fact is a list of accepted alternatives."""

    expect_insufficient: bool
    expected_ids: list[str] = Field(default_factory=list)
    key_facts: list[list[str]] = Field(default_factory=list, max_length=MAX_KEY_FACTS)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.expect_insufficient and (self.expected_ids or self.key_facts):
            raise ValueError("unanswerable cases carry no expected ids or facts")
        if not self.expect_insufficient and not (self.expected_ids and self.key_facts):
            raise ValueError("answerable cases need expected ids and 1-3 key facts")
        if any(not alternatives or not all(alternatives) for alternatives in self.key_facts):
            raise ValueError("every key fact needs at least one non-empty alternative")
        return self


class AnswerCase(_Strict):
    id: str
    suite: Literal["answer"]
    split: Split
    inputs: QuestionInputs
    labels: AnswerLabels


class DuplicateLabels(_Strict):
    """Expected verdict class (`duplicate` = duplicate or possible) and canonical original."""

    kind: Literal["labelled_duplicate", "hard_negative", "paraphrase", "novel"]
    expected: Literal["duplicate", "new"]
    original: str | None = None
    source_bug: str | None = None
    component: Component | None = None
    severity: Severity | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.expected == "duplicate") != (self.original is not None):
            raise ValueError("duplicate cases need an original; new cases must not have one")
        if self.kind == "novel" and (self.component is None or self.severity is None):
            raise ValueError("novel bugs are hand-labelled with component and severity")
        return self


class DuplicateCase(_Strict):
    id: str
    suite: Literal["duplicates"]
    split: Split
    inputs: ReportInputs
    labels: DuplicateLabels


class InjectionInputs(_Strict):
    """Either a question (`ask`) or a bug report (`bug_check`)."""

    mode: Literal["ask", "bug_check"]
    question: str | None = Field(default=None, min_length=1, max_length=1_000)
    title: str | None = Field(default=None, min_length=5, max_length=200)
    description: str | None = Field(default=None, min_length=1, max_length=4_000)
    platform: Platform | None = None
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.mode == "ask" and self.question is None:
            raise ValueError("ask cases need a question")
        report_fields = (self.title, self.description, self.platform, self.version)
        if self.mode == "bug_check" and any(value is None for value in report_fields):
            raise ValueError("bug_check cases need title, description, platform and version")
        return self

    def to_report(self) -> NewBugReport:
        """The submitted report (only valid for `bug_check` cases)."""
        if self.title is None or self.description is None or self.platform is None:
            raise ValueError("not a bug_check case")
        return NewBugReport(
            title=self.title,
            description=self.description,
            platform=self.platform,
            version=self.version or "",
        )


class InjectionLabels(_Strict):
    """What the case exercises and which outcomes are forbidden."""

    kind: Literal[
        "retrieved_injection",
        "duplicate_candidate_injection",
        "user_report_injection",
        "direct_user_injection",
    ]
    injected_id: str | None = None
    # `ask`: the doc the question must retrieve for the case to exercise the injection.
    expect_retrieved: str | None = None
    # ...and a marker the retrieved chunk must contain (multi-chunk docs).
    expect_chunk_contains: str | None = None
    # `bug_check`: the injected report that must be among the judged candidates.
    expect_candidate: str | None = None
    # True when the new report genuinely describes the injected report's defect, so a
    # `duplicate` verdict is legitimate if the score gate is met.
    genuine_match: bool = False
    forbidden_duplicate_of: list[str] = Field(default_factory=list)
    # Ids that must never appear in the answer text or sources.
    forbidden_ids: list[str] = Field(default_factory=list)


class InjectionCase(_Strict):
    id: str
    suite: Literal["injection"]
    split: Split
    inputs: InjectionInputs
    labels: InjectionLabels


class CaseSet(_Strict):
    """All case files, parsed."""

    retrieval: list[RetrievalCase]
    answer: list[AnswerCase]
    duplicates: list[DuplicateCase]
    injection: list[InjectionCase]
    # sha256 of each case file, recorded in the report so results map to a case version.
    fingerprints: dict[str, str]


def _read_jsonl[CaseT: BaseModel](path: Path, model: type[CaseT]) -> list[CaseT]:
    """Parse one JSONL file; errors name the file and line."""
    cases: list[CaseT] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(model.model_validate(json.loads(line)))
        except ValueError as exc:
            raise ValueError(f"{path.name}:{number}: invalid case: {exc}") from exc
    return cases


def load_cases(cases_dir: Path = CASES_DIR) -> CaseSet:
    """Load and validate every case file in `cases_dir`.

    Raises:
        OSError: if a file is missing.
        ValueError: if a line is not valid JSON, fails validation or reuses an id.
    """
    paths = {suite: cases_dir / name for suite, name in CASE_FILES.items()}
    case_set = CaseSet(
        retrieval=_read_jsonl(paths["retrieval"], RetrievalCase),
        answer=_read_jsonl(paths["answer"], AnswerCase),
        duplicates=_read_jsonl(paths["duplicates"], DuplicateCase),
        injection=_read_jsonl(paths["injection"], InjectionCase),
        fingerprints={
            name: hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            for name, path in ((p.name, p) for p in paths.values())
        },
    )
    ids = [
        case.id
        for group in (
            case_set.retrieval,
            case_set.answer,
            case_set.duplicates,
            case_set.injection,
        )
        for case in group
    ]
    repeated = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if repeated:
        raise ValueError(f"duplicate case ids: {repeated}")
    return case_set
