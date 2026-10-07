"""Pure functions that split dataset documents into `Chunk`s (no I/O).

- Bug report: one chunk (title, description, steps, expected/actual).
- Crash log: one chunk (summary, exception, top stack frames).
- Patch note: one summary chunk plus one chunk per fix.
- Markdown doc: one chunk per heading section, long sections split into overlapping
  character windows.
"""

import re
from collections.abc import Iterable

from studiodesk.data.loader import Dataset
from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import (
    BugReport,
    CrashLog,
    Doc,
    DocType,
    PatchNote,
    version_to_int,
)

MAX_STACK_FRAMES = 15
DOC_WINDOW_CHARS = 1_200
DOC_WINDOW_OVERLAP_CHARS = 150

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_bug_report(bug: BugReport) -> list[Chunk]:
    """Return the single chunk for a bug report."""
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(bug.steps_to_reproduce, 1))
    text = (
        f"{bug.description}\n\nSteps to reproduce:\n{steps}\n\n"
        f"Expected: {bug.expected}\nActual: {bug.actual}"
    )
    return [
        Chunk(
            doc_id=bug.id,
            doc_type=DocType.BUG_REPORT,
            chunk_index=0,
            title=bug.title,
            text=text,
            platform=[bug.platform],
            version=bug.version,
            version_num=version_to_int(bug.version),
            severity=bug.severity,
            component=bug.component,
            created_at=bug.created_at.isoformat(),
            duplicate_of=bug.duplicate_of,
        )
    ]


def top_stack_frames(stack_trace: str, limit: int = MAX_STACK_FRAMES) -> list[str]:
    """Return the first `limit` non-empty lines of a stack trace."""
    frames = [line.strip() for line in stack_trace.splitlines() if line.strip()]
    return frames[:limit]


def chunk_crash_log(crash: CrashLog, max_frames: int = MAX_STACK_FRAMES) -> list[Chunk]:
    """Return the single chunk for a crash log, keeping only the top `max_frames` frames."""
    frames = "\n".join(top_stack_frames(crash.stack_trace, max_frames))
    text = (
        f"{crash.summary}\n\nException: {crash.exception}\nBuild: {crash.build_hash}\n\n"
        f"Stack trace:\n{frames}"
    )
    return [
        Chunk(
            doc_id=crash.id,
            doc_type=DocType.CRASH_LOG,
            chunk_index=0,
            title=f"Crash: {crash.exception}",
            text=text,
            platform=[crash.platform],
            version=crash.version,
            version_num=version_to_int(crash.version),
            severity=crash.severity,
            component=crash.component,
            created_at=crash.created_at.isoformat(),
            related_bug=crash.related_bug,
        )
    ]


def chunk_patch_note(patch: PatchNote) -> list[Chunk]:
    """Return a summary chunk followed by one chunk per fix, in fix order."""
    common = {
        "doc_id": patch.id,
        "doc_type": DocType.PATCH_NOTE,
        "platform": list(patch.platforms),
        "version": patch.version,
        "version_num": version_to_int(patch.version),
        "created_at": patch.released_at.isoformat(),
    }
    parts = [patch.summary]
    if patch.fixes:
        parts.append("Fixed bugs: " + ", ".join(fix.bug_id for fix in patch.fixes))
    if patch.known_issues:
        parts.append("Known issues:\n" + "\n".join(f"- {i}" for i in patch.known_issues))
    chunks = [
        Chunk.model_validate(
            {
                **common,
                "chunk_index": 0,
                "title": f"Patch {patch.version} notes",
                "text": "\n\n".join(parts),
            }
        )
    ]
    for index, fix in enumerate(patch.fixes, 1):
        chunks.append(
            Chunk.model_validate(
                {
                    **common,
                    "chunk_index": index,
                    "title": f"Patch {patch.version}: fix for {fix.bug_id}",
                    "text": fix.description,
                    "related_bug": fix.bug_id,
                }
            )
        )
    return chunks


def split_markdown_sections(markdown: str) -> list[tuple[str | None, str]]:
    """Split markdown into `(heading, body)` pairs at every ATX heading line.

    Text before the first heading gets heading `None`. Bodies are stripped and may be empty.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            sections.append((match.group(2), []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections]


def window_text(text: str, size: int, overlap: int) -> list[str]:
    """Split `text` into windows of at most `size` chars, consecutive ones sharing `overlap`.

    Window ends snap back to the last whitespace, and later window starts snap forward to
    the next word, when possible, so words are not cut.
    Returns `[text]` when it already fits, and never returns empty windows.

    Raises:
        ValueError: unless `0 <= overlap < size`.
    """
    if not 0 <= overlap < size:
        raise ValueError("window_text requires 0 <= overlap < size")
    if len(text) <= size:
        return [text] if text.strip() else []
    windows: list[str] = []
    start = 0
    while True:
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(" ", start + overlap + 1, end), text.rfind("\n", start, end))
            if cut > start + overlap:
                end = cut
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end >= len(text):
            return windows
        start = _snap_start(text, end - overlap, end)


def _snap_start(text: str, start: int, end: int) -> int:
    """Move a window start that falls mid-word forward to the next word, staying < `end`.

    `start` is returned unchanged when it is already on a word boundary or when there is no
    whitespace in `[start, end - 1)` (hard cut). The result is always in `[start, end)`.
    """
    if start == 0 or text[start - 1].isspace() or text[start].isspace():
        return start
    for index in range(start, end - 1):
        if text[index].isspace():
            return index + 1
    return start


def chunk_doc(
    doc: Doc,
    window_chars: int = DOC_WINDOW_CHARS,
    overlap_chars: int = DOC_WINDOW_OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk a markdown doc by heading, then by overlapping windows; skips empty sections."""
    chunks: list[Chunk] = []
    for heading, body in split_markdown_sections(doc.body):
        title = doc.title if heading in (None, doc.title) else f"{doc.title} - {heading}"
        for window in window_text(body, window_chars, overlap_chars):
            chunks.append(
                Chunk(
                    doc_id=doc.id,
                    doc_type=DocType.DOC,
                    chunk_index=len(chunks),
                    title=title,
                    text=window,
                    platform=list(doc.platforms),
                )
            )
    return chunks


def chunk_dataset(dataset: Dataset) -> list[Chunk]:
    """Chunk every document of the dataset: bugs, crashes, patches, then docs."""
    groups: Iterable[list[Chunk]] = (
        *(chunk_bug_report(bug) for bug in dataset.bug_reports),
        *(chunk_crash_log(crash) for crash in dataset.crash_logs),
        *(chunk_patch_note(patch) for patch in dataset.patch_notes),
        *(chunk_doc(doc) for doc in dataset.docs),
    )
    return [chunk for group in groups for chunk in group]
