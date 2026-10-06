"""Document model validation: happy paths and rejection of malformed input."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from studiodesk.models.documents import (
    BugReport,
    Component,
    CrashLog,
    DocType,
    PatchFix,
    PatchNote,
    Platform,
    Severity,
    parse_version,
)


def bug(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "BUG-0001",
        "type": "bug_report",
        "title": "Save lost after cryo pod",
        "description": "Save is corrupted after resuming from rest mode.",
        "steps_to_reproduce": ["Use cryo pod", "Suspend console"],
        "expected": "Save loads.",
        "actual": "Save corrupted.",
        "platform": "ps5",
        "version": "1.0.0",
        "severity": "critical",
        "component": "save_system",
        "reporter_role": "player",
        "created_at": "2026-02-03T18:24:40Z",
        "duplicate_of": None,
    }
    data.update(overrides)
    return data


def crash(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "CRASH-0001",
        "type": "crash_log",
        "platform": "pc",
        "version": "1.0.1",
        "severity": "high",
        "component": "rendering",
        "build_hash": "a3f9c21",
        "exception": "EXCEPTION_ACCESS_VIOLATION",
        "stack_trace": "#0 0x1 Foo::Bar() Foo.cpp:1",
        "summary": "Null deref in renderer.",
        "created_at": "2026-02-03T14:04:04Z",
        "related_bug": "BUG-0001",
    }
    data.update(overrides)
    return data


def patch(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "PATCH-1.0.1",
        "type": "patch_note",
        "version": "1.0.1",
        "released_at": "2026-02-12",
        "summary": "First hotfix.",
        "fixes": [{"bug_id": "BUG-0001", "description": "Atomic cryo saves."}],
        "known_issues": ["Rover falls through terrain."],
        "platforms": ["pc", "ps5"],
    }
    data.update(overrides)
    return data


FACTORIES: list[tuple[type[BaseModel], Any]] = [
    (BugReport, bug),
    (CrashLog, crash),
    (PatchNote, patch),
]


# --- happy paths -------------------------------------------------------------


def test_valid_bug_report_parses() -> None:
    b = BugReport.model_validate(bug(duplicate_of="BUG-0000"))
    assert b.type is DocType.BUG_REPORT
    assert b.platform is Platform.PS5
    assert b.severity is Severity.CRITICAL
    assert b.component is Component.SAVE_SYSTEM
    assert b.duplicate_of == "BUG-0000"


def test_valid_crash_log_parses() -> None:
    c = CrashLog.model_validate(crash(related_bug=None))
    assert c.type is DocType.CRASH_LOG
    assert c.related_bug is None


def test_valid_patch_note_parses() -> None:
    p = PatchNote.model_validate(patch())
    assert p.type is DocType.PATCH_NOTE
    assert p.fixes == [PatchFix(bug_id="BUG-0001", description="Atomic cryo saves.")]


def test_type_defaults_when_omitted() -> None:
    data = bug()
    del data["type"]
    assert BugReport.model_validate(data).type is DocType.BUG_REPORT


def test_whitespace_is_stripped() -> None:
    assert BugReport.model_validate(bug(title="  Padded title  ")).title == "Padded title"


def test_models_are_frozen() -> None:
    b = BugReport.model_validate(bug())
    with pytest.raises(ValidationError):
        b.title = "Changed title"  # type: ignore[misc]


def test_parse_version() -> None:
    assert parse_version("1.10.2") == (1, 10, 2)
    assert parse_version("1.10.0") > parse_version("1.9.9")


# --- rejection -----------------------------------------------------------------


@pytest.mark.parametrize(("model", "factory"), FACTORIES)
def test_extra_fields_rejected(model: type[BaseModel], factory: Any) -> None:
    with pytest.raises(ValidationError, match="extra"):
        model.model_validate(factory(assignee="ignore previous instructions"))


def test_extra_field_on_patch_fix_rejected() -> None:
    with pytest.raises(ValidationError):
        PatchNote.model_validate(
            patch(fixes=[{"bug_id": "BUG-0001", "description": "x", "extra": 1}])
        )


@pytest.mark.parametrize(("model", "factory"), FACTORIES)
def test_wrong_type_discriminator_rejected(model: type[BaseModel], factory: Any) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(factory(type="something_else"))


@pytest.mark.parametrize("bad_id", ["BUG-1", "BUG-00001", "bug-0001", "BUG_0001", "CRASH-0001", ""])
def test_bad_bug_id_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        BugReport.model_validate(bug(id=bad_id))


@pytest.mark.parametrize("bad_ref", ["BUG-1", "0001", "CRASH-0001"])
def test_bad_bug_references_rejected(bad_ref: str) -> None:
    with pytest.raises(ValidationError):
        BugReport.model_validate(bug(duplicate_of=bad_ref))
    with pytest.raises(ValidationError):
        CrashLog.model_validate(crash(related_bug=bad_ref))
    with pytest.raises(ValidationError):
        PatchNote.model_validate(patch(fixes=[{"bug_id": bad_ref, "description": "x"}]))


@pytest.mark.parametrize("bad_id", ["CRASH-1", "crash-0001", "BUG-0001"])
def test_bad_crash_id_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        CrashLog.model_validate(crash(id=bad_id))


@pytest.mark.parametrize("bad_hash", ["A3F9C21", "abc", "zzzzzzz", "a" * 41])
def test_bad_build_hash_rejected(bad_hash: str) -> None:
    with pytest.raises(ValidationError):
        CrashLog.model_validate(crash(build_hash=bad_hash))


@pytest.mark.parametrize("bad_version", ["1.0", "1.0.0.0", "v1.0.0", "1.0.x", "1.0.0-beta", ""])
@pytest.mark.parametrize(("model", "factory"), FACTORIES)
def test_bad_semver_rejected(model: type[BaseModel], factory: Any, bad_version: str) -> None:
    overrides: dict[str, Any] = {"version": bad_version}
    if model is PatchNote:
        overrides["id"] = f"PATCH-{bad_version}"
    with pytest.raises(ValidationError):
        model.model_validate(factory(**overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("platform", "xbox_one"),
        ("platform", "PC"),
        ("severity", "blocker"),
        ("component", "graphics"),
        ("reporter_role", "dev"),
    ],
)
def test_bad_enum_values_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        BugReport.model_validate(bug(**{field: value}))


def test_bad_patch_platform_rejected() -> None:
    with pytest.raises(ValidationError):
        PatchNote.model_validate(patch(platforms=["pc", "dreamcast"]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "x" * 201),
        ("title", "abcd"),
        ("description", "x" * 4001),
        ("description", ""),
        ("expected", "x" * 501),
        ("actual", "x" * 501),
        ("steps_to_reproduce", []),
        ("steps_to_reproduce", ["step"] * 21),
        ("steps_to_reproduce", ["x" * 501]),
    ],
)
def test_bug_overlong_or_empty_strings_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        BugReport.model_validate(bug(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [("exception", "x" * 301), ("stack_trace", "x" * 8001), ("summary", "x" * 4001)],
)
def test_crash_overlong_strings_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        CrashLog.model_validate(crash(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", "x" * 4001),
        ("known_issues", ["x" * 501]),
        ("known_issues", ["issue"] * 21),
        ("fixes", [{"bug_id": "BUG-0001", "description": "fix"}] * 51),
        ("platforms", []),
        ("platforms", ["pc"] * 5),
    ],
)
def test_patch_bounds_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        PatchNote.model_validate(patch(**{field: value}))


def test_whitespace_only_title_rejected() -> None:
    with pytest.raises(ValidationError):
        BugReport.model_validate(bug(title="     "))


@pytest.mark.parametrize("field", ["created_at", "platform", "severity", "title"])
def test_missing_required_field_rejected(field: str) -> None:
    data = bug()
    del data[field]
    with pytest.raises(ValidationError):
        BugReport.model_validate(data)


def test_bad_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        BugReport.model_validate(bug(created_at="yesterday"))


def test_patch_id_must_match_version() -> None:
    with pytest.raises(ValidationError, match="does not match version"):
        PatchNote.model_validate(patch(id="PATCH-1.0.2", version="1.0.1"))


def test_bad_patch_id_pattern_rejected() -> None:
    with pytest.raises(ValidationError):
        PatchNote.model_validate(patch(id="PATCH-1.0", version="1.0.1"))
