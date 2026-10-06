"""Load and validate the synthetic Starfall Outpost dataset.

The dataset lives in a directory containing `bug_reports.json`, `crash_logs.json` and
`patch_notes.json`, each a JSON array of documents.
"""

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from studiodesk.models.documents import BugReport, CrashLog, PatchNote, parse_version

BUG_REPORTS_FILE = "bug_reports.json"
CRASH_LOGS_FILE = "crash_logs.json"
PATCH_NOTES_FILE = "patch_notes.json"

_BUGS = TypeAdapter(list[BugReport])
_CRASHES = TypeAdapter(list[CrashLog])
_PATCHES = TypeAdapter(list[PatchNote])


class Dataset(BaseModel):
    """All documents of the dataset, parsed and schema-validated."""

    model_config = ConfigDict(frozen=True)

    bug_reports: list[BugReport]
    crash_logs: list[CrashLog]
    patch_notes: list[PatchNote]


class ValidationReport(BaseModel):
    """Result of `validate_dataset`: per-type counts and any integrity errors."""

    counts: dict[str, int] = Field(default_factory=dict)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no errors were found."""
        return not self.errors


def load_bug_reports(path: Path) -> list[BugReport]:
    """Parse and validate a bug reports JSON file."""
    return _BUGS.validate_json(path.read_bytes())


def load_crash_logs(path: Path) -> list[CrashLog]:
    """Parse and validate a crash logs JSON file."""
    return _CRASHES.validate_json(path.read_bytes())


def load_patch_notes(path: Path) -> list[PatchNote]:
    """Parse and validate a patch notes JSON file."""
    return _PATCHES.validate_json(path.read_bytes())


def load_dataset(data_dir: Path) -> Dataset:
    """Load all three document files from `data_dir`.

    Raises:
        OSError: if a file is missing or unreadable.
        pydantic.ValidationError: if any record violates its schema.
    """
    return Dataset(
        bug_reports=load_bug_reports(data_dir / BUG_REPORTS_FILE),
        crash_logs=load_crash_logs(data_dir / CRASH_LOGS_FILE),
        patch_notes=load_patch_notes(data_dir / PATCH_NOTES_FILE),
    )


def _duplicate_id_errors(kind: str, ids: list[str]) -> list[str]:
    """Report ids that appear more than once."""
    return [f"{kind}: duplicate id {i} ({n}x)" for i, n in Counter(ids).items() if n > 1]


def _bug_reference_errors(dataset: Dataset) -> list[str]:
    """Check `duplicate_of` targets exist, differ from the bug and are earlier ids."""
    errors: list[str] = []
    bug_ids = {bug.id for bug in dataset.bug_reports}
    for bug in dataset.bug_reports:
        target = bug.duplicate_of
        if target is None:
            continue
        if target not in bug_ids:
            errors.append(f"{bug.id}: duplicate_of references unknown bug {target}")
        elif target >= bug.id:
            errors.append(f"{bug.id}: duplicate_of must reference an earlier bug, got {target}")
    return errors


def _crash_reference_errors(dataset: Dataset) -> list[str]:
    """Check `related_bug` targets exist."""
    bug_ids = {bug.id for bug in dataset.bug_reports}
    return [
        f"{crash.id}: related_bug references unknown bug {crash.related_bug}"
        for crash in dataset.crash_logs
        if crash.related_bug is not None and crash.related_bug not in bug_ids
    ]


def _patch_reference_errors(dataset: Dataset) -> list[str]:
    """Check patch fixes reference existing bugs reported on a version <= the patch version."""
    errors: list[str] = []
    bugs = {bug.id: bug for bug in dataset.bug_reports}
    for patch in dataset.patch_notes:
        patch_version = parse_version(patch.version)
        for fix in patch.fixes:
            bug = bugs.get(fix.bug_id)
            if bug is None:
                errors.append(f"{patch.id}: fix references unknown bug {fix.bug_id}")
            elif parse_version(bug.version) > patch_version:
                errors.append(f"{patch.id}: fixes {bug.id} reported on later version {bug.version}")
    return errors


def check_integrity(dataset: Dataset) -> list[str]:
    """Return all cross-record integrity errors for an already-parsed dataset."""
    return [
        *_duplicate_id_errors("bug_reports", [b.id for b in dataset.bug_reports]),
        *_duplicate_id_errors("crash_logs", [c.id for c in dataset.crash_logs]),
        *_duplicate_id_errors("patch_notes", [p.id for p in dataset.patch_notes]),
        *_bug_reference_errors(dataset),
        *_crash_reference_errors(dataset),
        *_patch_reference_errors(dataset),
    ]


def validate_dataset(data_dir: Path) -> ValidationReport:
    """Load the dataset in `data_dir` and report schema and integrity errors.

    Never raises for bad data: load failures are returned as errors in the report.
    """
    try:
        dataset = load_dataset(data_dir)
    except (OSError, ValidationError) as exc:
        return ValidationReport(errors=[f"failed to load dataset: {exc}"])

    severities: Counter[str] = Counter(b.severity.value for b in dataset.bug_reports)
    severities.update(c.severity.value for c in dataset.crash_logs)
    return ValidationReport(
        counts={
            "bug_reports": len(dataset.bug_reports),
            "crash_logs": len(dataset.crash_logs),
            "patch_notes": len(dataset.patch_notes),
        },
        severity_counts=dict(sorted(severities.items())),
        errors=check_integrity(dataset),
    )
