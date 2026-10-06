"""Integrity of the real synthetic dataset and detection of broken variants."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from studiodesk.data.loader import (
    BUG_REPORTS_FILE,
    CRASH_LOGS_FILE,
    PATCH_NOTES_FILE,
    Dataset,
    load_dataset,
    validate_dataset,
)
from studiodesk.models.documents import Component, Platform, parse_version

INJECTION_RE = re.compile(
    r"ignore (all )?(previous|prior) instructions|ignore the above|SYSTEM:", re.IGNORECASE
)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset(Path(__file__).resolve().parent.parent / "data" / "synthetic")


# --- real dataset ------------------------------------------------------------


def test_real_dataset_validates_cleanly(data_dir: Path) -> None:
    report = validate_dataset(data_dir)

    assert report.ok, report.errors
    assert report.errors == []
    assert (
        sum(report.counts.values())
        == sum(report.severity_counts.values()) + report.counts["patch_notes"]
    )


def test_total_count_around_100(dataset: Dataset) -> None:
    total = len(dataset.bug_reports) + len(dataset.crash_logs) + len(dataset.patch_notes)
    assert 90 <= total <= 110


def test_each_type_non_empty(dataset: Dataset) -> None:
    assert dataset.bug_reports
    assert dataset.crash_logs
    assert dataset.patch_notes


def test_at_least_ten_duplicates(dataset: Dataset) -> None:
    assert sum(1 for b in dataset.bug_reports if b.duplicate_of) >= 10


def test_every_platform_appears(dataset: Dataset) -> None:
    seen = {b.platform for b in dataset.bug_reports} | {c.platform for c in dataset.crash_logs}
    assert seen == set(Platform)


def test_every_component_appears(dataset: Dataset) -> None:
    seen = {b.component for b in dataset.bug_reports} | {c.component for c in dataset.crash_logs}
    assert seen == set(Component)


def test_ids_unique_across_files(dataset: Dataset) -> None:
    ids = (
        [b.id for b in dataset.bug_reports]
        + [c.id for c in dataset.crash_logs]
        + [p.id for p in dataset.patch_notes]
    )
    assert len(ids) == len(set(ids))


def test_duplicate_of_points_to_earlier_existing_bug(dataset: Dataset) -> None:
    by_id = {b.id: b for b in dataset.bug_reports}
    for b in dataset.bug_reports:
        if b.duplicate_of is None:
            continue
        assert b.duplicate_of in by_id, b.id
        assert b.duplicate_of < b.id, b.id


def test_related_bugs_exist(dataset: Dataset) -> None:
    bug_ids = {b.id for b in dataset.bug_reports}
    for c in dataset.crash_logs:
        assert c.related_bug is None or c.related_bug in bug_ids, c.id


def test_patch_fixes_reference_bugs_on_earlier_or_same_version(dataset: Dataset) -> None:
    by_id = {b.id: b for b in dataset.bug_reports}
    for p in dataset.patch_notes:
        for fix in p.fixes:
            assert fix.bug_id in by_id, (p.id, fix.bug_id)
            assert parse_version(by_id[fix.bug_id].version) <= parse_version(p.version)


def test_contains_prompt_injection_samples(dataset: Dataset) -> None:
    """The dataset must carry adversarial text so later milestones can test robustness."""
    hits = [b.id for b in dataset.bug_reports if INJECTION_RE.search(b.description)]
    assert len(hits) >= 3, hits


# --- broken variants ---------------------------------------------------------


def _read(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(path.read_text())
    return data


def _write(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(data))


def _dup_bug_id(d: Path) -> None:
    bugs = _read(d / BUG_REPORTS_FILE)
    bugs.append({**bugs[0]})
    _write(d / BUG_REPORTS_FILE, bugs)


def _dup_crash_id(d: Path) -> None:
    crashes = _read(d / CRASH_LOGS_FILE)
    crashes.append({**crashes[0]})
    _write(d / CRASH_LOGS_FILE, crashes)


def _dangling_duplicate_of(d: Path) -> None:
    bugs = _read(d / BUG_REPORTS_FILE)
    bugs[5]["duplicate_of"] = "BUG-0000"
    _write(d / BUG_REPORTS_FILE, bugs)


def _forward_duplicate_of(d: Path) -> None:
    bugs = _read(d / BUG_REPORTS_FILE)
    bugs[0]["duplicate_of"] = bugs[-1]["id"]
    _write(d / BUG_REPORTS_FILE, bugs)


def _self_duplicate_of(d: Path) -> None:
    bugs = _read(d / BUG_REPORTS_FILE)
    bugs[3]["duplicate_of"] = bugs[3]["id"]
    _write(d / BUG_REPORTS_FILE, bugs)


def _dangling_related_bug(d: Path) -> None:
    crashes = _read(d / CRASH_LOGS_FILE)
    crashes[0]["related_bug"] = "BUG-9999"
    _write(d / CRASH_LOGS_FILE, crashes)


def _dangling_patch_fix(d: Path) -> None:
    patches = _read(d / PATCH_NOTES_FILE)
    patches[0]["fixes"].append({"bug_id": "BUG-9999", "description": "Ghost fix."})
    _write(d / PATCH_NOTES_FILE, patches)


def _patch_fixes_future_bug(d: Path) -> None:
    bugs = _read(d / BUG_REPORTS_FILE)
    patches = _read(d / PATCH_NOTES_FILE)
    first = min(patches, key=lambda p: parse_version(p["version"]))
    latest = max(bugs, key=lambda b: parse_version(b["version"]))
    assert parse_version(latest["version"]) > parse_version(first["version"])
    first["fixes"].append({"bug_id": latest["id"], "description": "Time travel fix."})
    _write(d / PATCH_NOTES_FILE, patches)


BREAKERS: list[tuple[Callable[[Path], None], str]] = [
    (_dup_bug_id, "bug_reports: duplicate id BUG-0001"),
    (_dup_crash_id, "crash_logs: duplicate id CRASH-0001"),
    (_dangling_duplicate_of, "duplicate_of references unknown bug BUG-0000"),
    (_forward_duplicate_of, "duplicate_of must reference an earlier bug"),
    (_self_duplicate_of, "duplicate_of must reference an earlier bug"),
    (_dangling_related_bug, "related_bug references unknown bug BUG-9999"),
    (_dangling_patch_fix, "fix references unknown bug BUG-9999"),
    (_patch_fixes_future_bug, "reported on later version"),
]


@pytest.mark.parametrize(
    ("breaker", "expected"), BREAKERS, ids=[b.__name__.lstrip("_") for b, _ in BREAKERS]
)
def test_broken_dataset_reported(
    data_copy: Path, breaker: Callable[[Path], None], expected: str
) -> None:
    breaker(data_copy)

    report = validate_dataset(data_copy)

    assert not report.ok
    assert any(expected in e for e in report.errors), report.errors


def test_schema_violation_reported_not_raised(data_copy: Path) -> None:
    bugs = _read(data_copy / BUG_REPORTS_FILE)
    bugs[0]["severity"] = "apocalyptic"
    _write(data_copy / BUG_REPORTS_FILE, bugs)

    report = validate_dataset(data_copy)

    assert not report.ok
    assert report.errors[0].startswith("failed to load dataset")


def test_missing_file_reported_not_raised(data_copy: Path) -> None:
    (data_copy / PATCH_NOTES_FILE).unlink()

    report = validate_dataset(data_copy)

    assert not report.ok
    assert "failed to load dataset" in report.errors[0]


def test_malformed_json_reported_not_raised(data_copy: Path) -> None:
    (data_copy / CRASH_LOGS_FILE).write_text("[{not json")

    assert not validate_dataset(data_copy).ok


def test_missing_directory_reported(tmp_path: Path) -> None:
    assert not validate_dataset(tmp_path / "nope").ok
