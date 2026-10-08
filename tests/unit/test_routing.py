"""kNN routing: deterministic weighted votes, tie-breaking, self exclusion, allowlisted labels."""

import pytest

from fakes import FakeRetriever, scored
from studiodesk.actions.labels import LABEL_ALLOWLIST
from studiodesk.agent.retrieval import Retriever
from studiodesk.agent.routing import route_bug, route_neighbours, vote
from studiodesk.data.loader import Dataset
from studiodesk.embeddings import Embedder
from studiodesk.models.bugs import NewBugReport
from studiodesk.models.documents import Component, DocType, Severity
from studiodesk.vectorstore import QdrantStore

C = Component
S = Severity


def test_vote_weighted_by_score() -> None:
    winner, share = vote([(C.AUDIO, 0.9), (C.UI, 0.4), (C.UI, 0.4), (C.UI, 0.3)], list(Component))
    # UI weight 1.1 beats AUDIO 0.9 despite AUDIO having the best single score.
    assert winner is C.UI
    assert share == pytest.approx(1.1 / 2.0)


def test_vote_weight_tie_breaks_on_vote_count() -> None:
    winner, share = vote([(C.UI, 0.6), (C.AUDIO, 0.3), (C.AUDIO, 0.3)], list(Component))
    assert winner is C.AUDIO
    assert share == pytest.approx(0.5)


def test_vote_full_tie_breaks_on_enum_order() -> None:
    values = [(C.UI, 0.5), (C.RENDERING, 0.5)]
    assert vote(values, list(Component))[0] is C.RENDERING
    assert vote(list(reversed(values)), list(Component))[0] is C.RENDERING
    sev = [(S.LOW, 0.5), (S.CRITICAL, 0.5)]
    assert vote(sev, list(Severity))[0] is S.CRITICAL


def test_vote_ignores_none_and_negative_scores() -> None:
    winner, share = vote([(None, 0.99), (C.UI, -0.5), (C.AUDIO, 0.2)], list(Component))
    assert winner is C.AUDIO
    assert share == 1.0


def test_vote_empty_and_zero_weight() -> None:
    assert vote([], list(Component)) == (None, 0.0)
    assert vote([(None, 0.9)], list(Component)) == (None, 0.0)
    winner, share = vote([(C.UI, 0.0), (C.UI, -1.0), (C.AUDIO, 0.0)], list(Component))
    assert winner is C.UI  # more votes when all weights are zero
    assert share == 0.0


def test_vote_is_deterministic_under_permutation() -> None:
    values = [(C.UI, 0.5), (C.AUDIO, 0.5), (C.UI, 0.2), (C.AUDIO, 0.2), (C.INPUT, 0.7)]
    results = {
        vote(order, list(Component)) for order in (values, values[::-1], values[2:] + values[:2])
    }
    assert len(results) == 1


def test_route_neighbours_labels() -> None:
    neighbours = [
        scored("BUG-0001", 0.9, component=C.NETCODE, severity=S.HIGH),
        scored("BUG-0002", 0.8, component=C.NETCODE, severity=S.LOW),
        scored("BUG-0003", 0.7, component=C.UI, severity=S.HIGH),
    ]

    result = route_neighbours(neighbours)

    assert result.component is C.NETCODE
    assert result.severity is S.HIGH
    assert result.labels == ["bug", "component:netcode", "severity:high"]
    assert result.neighbours == ["BUG-0001", "BUG-0002", "BUG-0003"]
    assert set(result.labels) <= LABEL_ALLOWLIST


def test_route_neighbours_without_metadata_only_base_label() -> None:
    result = route_neighbours([scored("DOC-x", 0.9, doc_type=DocType.DOC)])
    assert result.labels == ["bug"]
    assert result.component is None and result.severity is None
    assert route_neighbours([]).labels == ["bug"]


def test_route_bug_uses_bug_reports_only_and_excludes_ids() -> None:
    retriever = FakeRetriever(hits=[scored("BUG-0001", 0.9, component=C.UI, severity=S.LOW)])
    report = NewBugReport(title="Title here", description="d", platform="pc", version="1.0.0")

    result = route_bug(report, retriever, k=3, exclude_ids={"BUG-0001"})

    assert retriever.searches[0]["filters"] == "bug_reports_only"
    assert retriever.searches[0]["top_k"] == 3
    assert result.neighbours == []
    assert result.labels == ["bug"]


def test_route_every_dataset_bug_deterministically(
    ingested_store: QdrantStore, fake_embedder: Embedder, dataset: Dataset
) -> None:
    retriever = Retriever(fake_embedder, ingested_store)
    for bug in dataset.bug_reports:
        report = NewBugReport(
            title=bug.title,
            description=bug.description,
            platform=bug.platform,
            version=bug.version,
        )
        first = route_bug(report, retriever, k=7, exclude_ids={bug.id})
        second = route_bug(report, retriever, k=7, exclude_ids={bug.id})

        assert first == second
        assert bug.id not in first.neighbours
        assert len(first.neighbours) == 7
        assert all(n.startswith("BUG-") for n in first.neighbours)
        assert set(first.labels) <= LABEL_ALLOWLIST
        assert first.labels[0] == "bug"
        assert first.component is not None and first.severity is not None
