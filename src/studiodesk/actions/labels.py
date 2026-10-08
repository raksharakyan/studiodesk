"""The fixed allowlist of GitHub labels StudioDesk may put on issues.

Labels are derived only from enums (component, severity) plus a fixed base label, so no
request or model text can introduce a label. `scripts/setup_issue_labels.py` creates
exactly these labels in the target repository.
"""

from collections.abc import Iterable
from typing import NamedTuple

from studiodesk.models.documents import Component, Severity

BASE_LABEL = "bug"

_SEVERITY_COLOURS = {
    Severity.CRITICAL: "b60205",
    Severity.HIGH: "d93f0b",
    Severity.MEDIUM: "fbca04",
    Severity.LOW: "0e8a16",
}
_COMPONENT_COLOUR = "1d76db"
_BASE_COLOUR = "d73a4a"


class LabelSpec(NamedTuple):
    """A label as created in the repository."""

    name: str
    color: str
    description: str


def component_label(component: Component) -> str:
    """Label for a game component, e.g. `component:netcode`."""
    return f"component:{component.value}"


def severity_label(severity: Severity) -> str:
    """Label for a severity, e.g. `severity:high`."""
    return f"severity:{severity.value}"


LABEL_SPECS: tuple[LabelSpec, ...] = (
    LabelSpec(BASE_LABEL, _BASE_COLOUR, "Something isn't working"),
    *(LabelSpec(component_label(c), _COMPONENT_COLOUR, f"Component: {c.value}") for c in Component),
    *(LabelSpec(severity_label(s), _SEVERITY_COLOURS[s], f"Severity: {s.value}") for s in Severity),
)
LABEL_ALLOWLIST: frozenset[str] = frozenset(spec.name for spec in LABEL_SPECS)


def allowed_labels(labels: Iterable[str]) -> list[str]:
    """Keep only allowlisted labels, deduplicated, in first-seen order."""
    seen: dict[str, None] = {}
    for label in labels:
        if label in LABEL_ALLOWLIST:
            seen.setdefault(label, None)
    return list(seen)
