"""Pure metric functions used by the eval suites (no I/O, no randomness).

Conventions:
- Ranked lists are doc ids in rank order; `dedupe_ranked` keeps the first occurrence, so a
  document split into several chunks counts once at its best rank.
- Division by zero yields 0.0 (an empty set of predictions has precision 0, not NaN).
- Macro-F1 averages per-class F1 over the classes present in either `y_true` or `y_pred`
  (the scikit-learn default), so a class that is predicted but never true still counts.
"""

from collections.abc import Collection, Hashable, Iterable, Sequence
from dataclasses import dataclass


def dedupe_ranked(ids: Iterable[str]) -> list[str]:
    """Drop repeated ids, keeping the first (best-ranked) occurrence."""
    return list(dict.fromkeys(ids))


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any relevant id is among the first `k` ranked ids, else 0.0."""
    return 1.0 if any(doc_id in relevant for doc_id in ranked[:k]) else 0.0


def reciprocal_rank_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / rank of the first relevant id within the first `k`, else 0.0."""
    for rank, doc_id in enumerate(ranked[:k], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the relevant ids found among the first `k` ranked ids."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean; 0.0 for no values."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, or 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class BinaryCounts:
    """Confusion counts for a binary decision."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return f1_score(self.precision, self.recall)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (0.0 when both are 0)."""
    return safe_div(2 * precision * recall, precision + recall)


def binary_counts(y_true: Sequence[bool], y_pred: Sequence[bool]) -> BinaryCounts:
    """Count TP/FP/FN/TN for paired boolean labels and predictions.

    Raises:
        ValueError: if the sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    pairs = list(zip(y_true, y_pred, strict=True))
    return BinaryCounts(
        tp=sum(t and p for t, p in pairs),
        fp=sum(p and not t for t, p in pairs),
        fn=sum(t and not p for t, p in pairs),
        tn=sum(not t and not p for t, p in pairs),
    )


def precision_recall_f1(
    y_true: Sequence[bool], y_pred: Sequence[bool]
) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class."""
    counts = binary_counts(y_true, y_pred)
    return counts.precision, counts.recall, counts.f1


def accuracy[T: Hashable](y_true: Sequence[T], y_pred: Sequence[T]) -> float:
    """Share of positions where prediction equals label.

    Raises:
        ValueError: if the sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return safe_div(sum(t == p for t, p in zip(y_true, y_pred, strict=True)), len(y_true))


def class_labels[T: Hashable](y_true: Sequence[T], y_pred: Sequence[T]) -> list[T]:
    """Classes present in either sequence, in first-seen order (labels first)."""
    return list(dict.fromkeys([*y_true, *y_pred]))


def per_class_f1[T: Hashable](
    y_true: Sequence[T], y_pred: Sequence[T], labels: Sequence[T] | None = None
) -> dict[T, float]:
    """One-vs-rest F1 for each class."""
    classes = list(labels) if labels is not None else class_labels(y_true, y_pred)
    return {
        label: binary_counts([t == label for t in y_true], [p == label for p in y_pred]).f1
        for label in classes
    }


def macro_f1[T: Hashable](
    y_true: Sequence[T], y_pred: Sequence[T], labels: Sequence[T] | None = None
) -> float:
    """Unweighted mean of per-class F1 (see the module docstring for the class set).

    Raises:
        ValueError: if the sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return mean(per_class_f1(y_true, y_pred, labels).values())


def confusion_matrix[T: Hashable](
    y_true: Sequence[T], y_pred: Sequence[T], labels: Sequence[T]
) -> list[list[int]]:
    """`matrix[i][j]` = number of cases with true `labels[i]` predicted as `labels[j]`.

    Pairs whose label or prediction is not in `labels` are not counted.

    Raises:
        ValueError: if the sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred, strict=True):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix
