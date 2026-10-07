"""Pure chunking functions: one chunk per bug/crash, summary + fixes per patch, docs by heading."""

import uuid
from datetime import UTC, date, datetime

import pytest

from studiodesk.data.loader import Dataset
from studiodesk.ingest.chunking import (
    DOC_WINDOW_CHARS,
    DOC_WINDOW_OVERLAP_CHARS,
    MAX_STACK_FRAMES,
    chunk_bug_report,
    chunk_crash_log,
    chunk_dataset,
    chunk_doc,
    chunk_patch_note,
    split_markdown_sections,
    top_stack_frames,
    window_text,
)
from studiodesk.models.chunks import Chunk, chunk_point_id
from studiodesk.models.documents import (
    BugReport,
    Component,
    CrashLog,
    Doc,
    DocType,
    PatchFix,
    PatchNote,
    Platform,
    Severity,
    version_to_int,
)


def _bug(**overrides: object) -> BugReport:
    data: dict[str, object] = {
        "id": "BUG-0100",
        "title": "Cryo pod eats my save",
        "description": "After sleeping in the cryo pod the save is gone.",
        "steps_to_reproduce": ["Sleep in cryo pod", "Quit to menu", "Load save"],
        "expected": "Save loads.",
        "actual": "Save is corrupted.",
        "platform": "ps5",
        "version": "1.2.3",
        "severity": "high",
        "component": "save_system",
        "reporter_role": "player",
        "created_at": datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        "duplicate_of": "BUG-0001",
    }
    data.update(overrides)
    return BugReport.model_validate(data)


def _crash(frames: int = 30) -> CrashLog:
    trace = "\n".join(f"  at Frame{i}()" for i in range(frames))
    return CrashLog.model_validate(
        {
            "id": "CRASH-0100",
            "platform": "switch",
            "version": "1.1.0",
            "severity": "critical",
            "component": "rendering",
            "build_hash": "abc1234",
            "exception": "NullReferenceException",
            "stack_trace": trace,
            "summary": "Crash when docking.",
            "created_at": datetime(2026, 2, 1, tzinfo=UTC),
            "related_bug": "BUG-0007",
        }
    )


def _patch(fixes: int = 2, known_issues: list[str] | None = None) -> PatchNote:
    return PatchNote(
        id="PATCH-1.4.0",
        version="1.4.0",
        released_at=date(2026, 6, 1),
        summary="Big stability update.",
        fixes=[
            PatchFix(bug_id=f"BUG-{i:04d}", description=f"Fixed thing {i}.")
            for i in range(1, fixes + 1)
        ],
        known_issues=known_issues or [],
        platforms=[Platform.PC, Platform.PS5],
    )


def _doc(body: str) -> Doc:
    return Doc(id="DOC-test", title="Test Guide", source="test.md", body=body)


# --- bug reports -------------------------------------------------------------------------


def test_bug_report_is_one_chunk_with_all_fields() -> None:
    bug = _bug()

    [chunk] = chunk_bug_report(bug)

    assert chunk.doc_id == "BUG-0100"
    assert chunk.doc_type is DocType.BUG_REPORT
    assert chunk.chunk_index == 0
    assert chunk.title == bug.title
    assert bug.description in chunk.text
    assert "1. Sleep in cryo pod\n2. Quit to menu\n3. Load save" in chunk.text
    assert "Expected: Save loads." in chunk.text
    assert "Actual: Save is corrupted." in chunk.text
    assert chunk.platform == [Platform.PS5]
    assert chunk.version == "1.2.3"
    assert chunk.version_num == 10203
    assert chunk.severity is Severity.HIGH
    assert chunk.component is Component.SAVE_SYSTEM
    assert chunk.created_at is not None
    assert chunk.created_at.startswith("2026-03-01T12:00:00")
    assert chunk.duplicate_of == "BUG-0001"
    assert chunk.related_bug is None


def test_bug_without_duplicate_has_none() -> None:
    [chunk] = chunk_bug_report(_bug(duplicate_of=None))

    assert chunk.duplicate_of is None


# --- crash logs --------------------------------------------------------------------------


def test_crash_log_keeps_only_top_frames() -> None:
    [chunk] = chunk_crash_log(_crash(frames=30))

    assert chunk.doc_type is DocType.CRASH_LOG
    assert chunk.title == "Crash: NullReferenceException"
    assert "at Frame0()" in chunk.text
    assert f"at Frame{MAX_STACK_FRAMES - 1}()" in chunk.text
    assert f"at Frame{MAX_STACK_FRAMES}()" not in chunk.text
    assert "Build: abc1234" in chunk.text
    assert chunk.related_bug == "BUG-0007"
    assert chunk.duplicate_of is None
    assert chunk.platform == [Platform.SWITCH]
    assert chunk.version_num == version_to_int("1.1.0")


def test_crash_log_with_few_frames_keeps_all() -> None:
    [chunk] = chunk_crash_log(_crash(frames=3), max_frames=15)

    assert all(f"at Frame{i}()" in chunk.text for i in range(3))


def test_top_stack_frames_skips_blank_lines_and_strips() -> None:
    assert top_stack_frames("\n  a  \n\n b\n   \nc", limit=2) == ["a", "b"]


# --- patch notes -------------------------------------------------------------------------


def test_patch_note_summary_plus_one_chunk_per_fix() -> None:
    chunks = chunk_patch_note(_patch(fixes=3, known_issues=["Rare audio pop"]))

    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    summary, *fixes = chunks
    assert summary.title == "Patch 1.4.0 notes"
    assert "Big stability update." in summary.text
    assert "Fixed bugs: BUG-0001, BUG-0002, BUG-0003" in summary.text
    assert "- Rare audio pop" in summary.text
    assert summary.related_bug is None
    assert [f.related_bug for f in fixes] == ["BUG-0001", "BUG-0002", "BUG-0003"]
    assert [f.text for f in fixes] == ["Fixed thing 1.", "Fixed thing 2.", "Fixed thing 3."]
    for chunk in chunks:
        assert chunk.doc_id == "PATCH-1.4.0"
        assert chunk.doc_type is DocType.PATCH_NOTE
        assert chunk.platform == [Platform.PC, Platform.PS5]
        assert chunk.version_num == 10400
        assert chunk.severity is None


def test_patch_note_without_fixes_is_summary_only() -> None:
    [summary] = chunk_patch_note(_patch(fixes=0))

    assert summary.text == "Big stability update."


# --- markdown docs -----------------------------------------------------------------------


def test_split_markdown_sections() -> None:
    md = "intro line\n# Title\n\nbody one\n## Sub ##\nbody two\n###not-a-heading\n"

    assert split_markdown_sections(md) == [
        (None, "intro line"),
        ("Title", "body one"),
        ("Sub", "body two\n###not-a-heading"),
    ]


def test_doc_chunked_by_heading_with_titles_and_no_empty_chunks() -> None:
    body = "# Test Guide\n\nIntro text.\n\n## Saving\n\nSave often.\n\n## Empty\n\n## Audio\nLoud."

    chunks = chunk_doc(_doc(body))

    assert [(c.title, c.text) for c in chunks] == [
        ("Test Guide", "Intro text."),
        ("Test Guide - Saving", "Save often."),
        ("Test Guide - Audio", "Loud."),
    ]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.doc_type is DocType.DOC and c.version_num is None for c in chunks)
    assert chunks[0].platform == list(Platform)


def test_long_doc_section_split_into_overlapping_windows() -> None:
    words = " ".join(f"word{i:04d}" for i in range(600))  # ~5,400 chars
    chunks = chunk_doc(_doc(f"# Test Guide\n\n## Long\n{words}"))

    assert len(chunks) > 1
    assert all(len(c.text) <= DOC_WINDOW_CHARS for c in chunks)
    assert all(c.title == "Test Guide - Long" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Windows end on a whole word, every word is fully contained in some window, and
    # consecutive windows overlap (the previous window's last word reappears in the next).
    vocab = set(words.split())
    assert all(c.text.split()[-1] in vocab for c in chunks)
    assert all(any(word in c.text for c in chunks) for word in vocab)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.text.split()[-1] in nxt.text


def test_windows_do_not_start_mid_word() -> None:
    words = " ".join(f"word{i:04d}" for i in range(600))
    vocab = set(words.split())

    windows = window_text(words, size=DOC_WINDOW_CHARS, overlap=DOC_WINDOW_OVERLAP_CHARS)

    assert all(w.split()[0] in vocab for w in windows)


def test_window_text_overlap_and_bounds() -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))  # no whitespace: hard cuts

    windows = window_text(text, size=300, overlap=50)

    assert all(0 < len(w) <= 300 for w in windows)
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert prev[-50:] == nxt[:50]
    assert windows[0] == text[:300]
    assert text.endswith(windows[-1])


def test_window_text_snaps_to_whitespace() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"

    windows = window_text(text, size=20, overlap=5)

    vocab = text.split()
    for window in windows:
        assert window == window.strip()
        assert len(window) <= 20
        assert window.split()[-1] in vocab, "window ends on a whole word"
    assert windows[-1].endswith("kappa")


@pytest.mark.parametrize(("text", "expected"), [("short", ["short"]), ("   \n ", []), ("", [])])
def test_window_text_small_inputs(text: str, expected: list[str]) -> None:
    assert window_text(text, size=DOC_WINDOW_CHARS, overlap=DOC_WINDOW_OVERLAP_CHARS) == expected


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (100, -1), (0, 0)])
def test_window_text_rejects_bad_overlap(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        window_text("x" * 500, size=size, overlap=overlap)


# --- whole dataset -----------------------------------------------------------------------


def test_chunk_dataset_counts_and_invariants(dataset: Dataset) -> None:
    chunks = chunk_dataset(dataset)

    by_type = {t: [c for c in chunks if c.doc_type is t] for t in DocType}
    assert len(by_type[DocType.BUG_REPORT]) == len(dataset.bug_reports)
    assert len(by_type[DocType.CRASH_LOG]) == len(dataset.crash_logs)
    expected_patch_chunks = sum(1 + len(p.fixes) for p in dataset.patch_notes)
    assert len(by_type[DocType.PATCH_NOTE]) == expected_patch_chunks
    assert {c.doc_id for c in by_type[DocType.DOC]} == {d.id for d in dataset.docs}
    # Each doc yields at least one chunk per non-empty section.
    for doc in dataset.docs:
        sections = [b for _, b in split_markdown_sections(doc.body) if b.strip()]
        assert len([c for c in by_type[DocType.DOC] if c.doc_id == doc.id]) >= len(sections)

    assert all(c.text.strip() for c in chunks), "no empty chunks"
    point_ids = [c.point_id for c in chunks]
    assert len(set(point_ids)) == len(point_ids), "point ids are unique"
    # Order: bugs, crashes, patches, docs.
    order = [c.doc_type for c in chunks]
    assert order == sorted(order, key=list(DocType).index)


def test_chunk_dataset_is_deterministic(dataset: Dataset) -> None:
    assert chunk_dataset(dataset) == chunk_dataset(dataset)


# --- chunk model / version packing -------------------------------------------------------


def test_point_id_is_deterministic_uuid5() -> None:
    pid = chunk_point_id("BUG-0001", 0)

    assert uuid.UUID(pid).version == 5
    assert pid == chunk_point_id("BUG-0001", 0)
    assert pid != chunk_point_id("BUG-0001", 1)
    assert pid != chunk_point_id("BUG-0002", 0)
    chunk = Chunk(
        doc_id="BUG-0001", doc_type=DocType.BUG_REPORT, chunk_index=0, title="t", text="x"
    )
    assert chunk.point_id == pid
    assert chunk.embedding_text == "t\nx"


@pytest.mark.parametrize(
    ("lower", "higher"),
    [("1.0.0", "1.0.1"), ("1.0.9", "1.1.0"), ("1.9.9", "1.10.0"), ("1.99.99", "2.0.0")],
)
def test_version_to_int_orders_like_semver(lower: str, higher: str) -> None:
    assert version_to_int(lower) < version_to_int(higher)


def test_version_to_int_packing() -> None:
    assert version_to_int("1.2.3") == 10203
    assert version_to_int("0.0.0") == 0


@pytest.mark.parametrize("bad", ["1.100.0", "100.0.0", "1.0.100"])
def test_version_to_int_rejects_parts_that_break_ordering(bad: str) -> None:
    with pytest.raises(ValueError, match="< 100"):
        version_to_int(bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": ""},
        {"chunk_index": -1},
        {"duplicate_of": "not-a-bug"},
        {"unexpected": "field"},
        {"doc_type": "tweet"},
    ],
)
def test_chunk_model_rejects_invalid(overrides: dict[str, object]) -> None:
    from pydantic import ValidationError

    data = {"doc_id": "X", "doc_type": "doc", "chunk_index": 0, "title": "t", "text": "x"}
    with pytest.raises(ValidationError):
        Chunk.model_validate({**data, **overrides})
