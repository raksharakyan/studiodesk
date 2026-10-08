"""Build the GitHub issue for a submitted report, server-side, from fixed fields only.

The issue text comes only from the user's validated report fields and the kNN routing
result. No model output is ever included, so the LLM cannot put content into an outward
action. `@mentions` are neutralised with a zero-width joiner so filing an issue never
pings users or teams.
"""

import re

from studiodesk.models.actions import ISSUE_BODY_MAX_CHARS, ISSUE_TITLE_MAX_CHARS, IssueDraft
from studiodesk.models.bugs import NewBugReport, RoutingResult

ZERO_WIDTH_JOINER = "‍"
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9])")
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


def build_issue_body(report: NewBugReport, routing: RoutingResult) -> str:
    """Render the markdown body from report fields plus routing, capped in length."""
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
        "",
        FOOTER,
    ]
    body = neutralise_mentions("\n".join(lines))
    if len(body) > ISSUE_BODY_MAX_CHARS:
        body = body[: ISSUE_BODY_MAX_CHARS - len(_TRUNCATED_MARK)] + _TRUNCATED_MARK
    return body


def build_issue_draft(report: NewBugReport, routing: RoutingResult) -> IssueDraft:
    """Build the exact issue (title, body, allowlisted labels) that confirm will create."""
    title = neutralise_mentions(report.title)[:ISSUE_TITLE_MAX_CHARS]
    return IssueDraft(
        title=title, body=build_issue_body(report, routing), labels=list(routing.labels)
    )
