"""Ticket routing: predict component and severity by a kNN vote over similar bug reports.

No LLM is involved. The `k` most similar bug reports vote with weight equal to their
(non-negative) similarity score; the winner's share is its weight over the total weight.
Ties break on more votes, then on enum declaration order, so results are deterministic.
Predicted values map to labels from the fixed allowlist only.
"""

from collections.abc import Collection, Sequence
from enum import StrEnum

from studiodesk.actions.labels import (
    BASE_LABEL,
    allowed_labels,
    component_label,
    severity_label,
)
from studiodesk.agent.retrieval import Retriever
from studiodesk.models.bugs import NewBugReport, RoutingResult
from studiodesk.models.documents import Component, Severity
from studiodesk.vectorstore import ScoredChunk


def vote[E: StrEnum](
    values: Sequence[tuple[E | None, float]], order: Sequence[E]
) -> tuple[E | None, float]:
    """Return the score-weighted winner among non-None values and its share of the weight.

    Args:
        values: `(value, score)` per neighbour; None values and negative scores do not vote.
        order: All enum members in declaration order, used as the final tie-breaker.
    """
    weights: dict[E, float] = {}
    counts: dict[E, int] = {}
    for value, score in values:
        if value is None:
            continue
        weights[value] = weights.get(value, 0.0) + max(score, 0.0)
        counts[value] = counts.get(value, 0) + 1
    total = sum(weights.values())
    if not weights:
        return None, 0.0
    winner = min(weights, key=lambda v: (-weights[v], -counts[v], order.index(v)))
    share = weights[winner] / total if total > 0 else 0.0
    return winner, min(share, 1.0)


def route_neighbours(neighbours: list[ScoredChunk]) -> RoutingResult:
    """Vote component and severity over `neighbours` and derive allowlisted labels."""
    component, component_share = vote(
        [(n.chunk.component, n.score) for n in neighbours], list(Component)
    )
    severity, severity_share = vote(
        [(n.chunk.severity, n.score) for n in neighbours], list(Severity)
    )
    labels = [BASE_LABEL]
    if component is not None:
        labels.append(component_label(component))
    if severity is not None:
        labels.append(severity_label(severity))
    return RoutingResult(
        component=component,
        component_share=round(component_share, 4),
        severity=severity,
        severity_share=round(severity_share, 4),
        neighbours=[n.chunk.doc_id for n in neighbours],
        labels=allowed_labels(labels),
    )


def route_bug(
    report: NewBugReport, retriever: Retriever, *, k: int, exclude_ids: Collection[str] = ()
) -> RoutingResult:
    """Route `report` by its `k` nearest bug reports, never counting `exclude_ids`.

    Pass the report's own id in `exclude_ids` when routing a report that is already in the
    dataset (evaluation), so it does not vote for itself.

    Raises:
        VectorStoreError: if retrieval fails.
    """
    neighbours = retriever.search_bug_reports(report.search_text, k, exclude_ids=exclude_ids)
    return route_neighbours(neighbours)
