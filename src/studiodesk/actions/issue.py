"""Build the GitHub issue for a submitted report, server-side, from fixed fields only.

The issue text comes only from the user's validated report fields, the kNN routing result
and, for a `possible_duplicate` verdict, the ids and similarity scores of the flagged
candidates from our own retrieval. No model text (answers, reasons) is ever included, so
the LLM cannot put content into an outward action. `@mentions` are neutralised with a
zero-width joiner so filing an issue never pings users or teams.
"""

import re

from studiodesk.actions.labels import POSSIBLE_DUPLICATE_LABEL, allowed_labels
from studiodesk.models.actions import ISSUE_BODY_MAX_CHARS, ISSUE_TITLE_MAX_CHARS, IssueDraft
from studiodesk.models.bugs import (
    DuplicateCandidate,
    DuplicateCheck,
    DuplicateVerdict,
    NewBugReport,
    RoutingResult,
)
from studiodesk.models.documents import BUG_ID_PATTERN

ZERO_WIDTH_JOINER = "‍"
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9])")
_BUG_ID_RE = re.compile(BUG_ID_PATTERN)
MAX_POSSIBLE_DUPLICATES = 5
_TRUNCATED_MARK = "\n\n_(truncated)_"
FOOTER = "_Filed by StudioDesk after explicit user confirmation._"


def neutralise_mentions(text: str) -> str:
    """Insert a zero-width joiner after every `@` that starts a mention (`@user`, `@org/x`)."""
    return _MENTION_RE.sub("@" + ZERO_WIDTH_JOINER, text)


def _routing_lines(routing: RoutingResult) -> list[str]:
    """Markdown lines describing the automatic triage."""
    component = routing.component.value if routing.component else "unknown"
    severity = routing.severity.value if routing.severity else "unknown"
    lines = [
        "## Triage (automatic, kNN over similar reports)",
        f"- Component: {component} (vote share {routing.component_share:.2f})",
        f"- Severity: {severity} (vote share {routing.severity_share:.2f})",
    ]
    if routing.neighbours:
        lines.append(f"- Similar reports: {', '.join(routing.neighbours)}")
    return lines


def flagged_candidates(duplicates: DuplicateCheck) -> list[DuplicateCandidate]:
    """Candidates behind a `possible_duplicate` verdict, best score first (bug ids only)."""
    if duplicates.verdict is not DuplicateVerdict.POSSIBLE_DUPLICATE:
        return []
    flagged = [
        c
        for c in duplicates.candidates
        if (c.is_duplicate or c.score >= duplicates.auto_threshold)
        and _BUG_ID_RE.fullmatch(c.doc_id)
    ]
    return sorted(flagged, key=lambda c: c.score, reverse=True)[:MAX_POSSIBLE_DUPLICATES]


def possible_duplicates_line(candidates: list[DuplicateCandidate]) -> str:
    """Server-written line, e.g. `Possible duplicates: BUG-0002 (score 0.62)`."""
    return "Possible duplicates: " + ", ".join(
        f"{c.doc_id} (score {c.score:.2f})" for c in candidates
    )


def build_issue_body(
    report: NewBugReport, routing: RoutingResult, possible: list[DuplicateCandidate]
) -> str:
    """Render the markdown body from report fields, routing and flagged ids, capped."""
    lines = ["## Description", report.description, ""]
    if report.steps_to_reproduce:
        lines.append("## Steps to reproduce")
        lines += [f"{i}. {step}" for i, step in enumerate(report.steps_to_reproduce, 1)]
        lines.append("")
    if report.expected:
        lines += ["## Expected", report.expected, ""]
    if report.actual:
        lines += ["## Actual", report.actual, ""]
    lines += [
        "## Environment",
        f"- Platform: {report.platform.value}",
        f"- Version: {report.version}",
        "",
        *_routing_lines(routing),
        *([possible_duplicates_line(possible)] if possible else []),
        "",
        FOOTER,
    ]
    body = neutralise_mentions("\n".join(lines))
    if len(body) > ISSUE_BODY_MAX_CHARS:
        body = body[: ISSUE_BODY_MAX_CHARS - len(_TRUNCATED_MARK)] + _TRUNCATED_MARK
    return body


def build_issue_draft(
    report: NewBugReport, routing: RoutingResult, duplicates: DuplicateCheck
) -> IssueDraft:
    """Build the exact issue (title, body, allowlisted labels) that confirm will create."""
    title = neutralise_mentions(report.title)[:ISSUE_TITLE_MAX_CHARS]
    possible = flagged_candidates(duplicates)
    labels = list(routing.labels)
    if duplicates.verdict is DuplicateVerdict.POSSIBLE_DUPLICATE:
        labels.append(POSSIBLE_DUPLICATE_LABEL)
    return IssueDraft(
        title=title,
        body=build_issue_body(report, routing, possible),
        labels=allowed_labels(labels),
    )
