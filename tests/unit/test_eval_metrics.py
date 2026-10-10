"""Eval metric functions against hand-computed values."""

import pytest
from evals import metrics

RANKED = ["A", "B", "C", "D", "E", "F"]
RELEVANT = {"C", "F", "Z"}


def test_dedupe_ranked_keeps_first_occurrence() -> None:
    assert metrics.dedupe_ranked(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]
    assert metrics.dedupe_ranked([]) == []


@pytest.mark.parametrize(("k", "expected"), [(5, 1.0), (3, 1.0), (2, 0.0), (0, 0.0)])
def test_hit_at_k(k: int, expected: float) -> None:
    assert metrics.hit_at_k(RANKED, RELEVANT, k) == expected


@pytest.mark.parametrize(("k", "expected"), [(5, 1 / 3), (3, 1 / 3), (2, 0.0)])
def test_reciprocal_rank_at_k(k: int, expected: float) -> None:
    assert metrics.reciprocal_rank_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)


def test_reciprocal_rank_first_position_is_one() -> None:
    assert metrics.reciprocal_rank_at_k(["C", "A"], RELEVANT, 5) == 1.0


@pytest.mark.parametrize(("k", "expected"), [(5, 1 / 3), (6, 2 / 3), (2, 0.0)])
def test_recall_at_k(k: int, expected: float) -> None:
    # C (rank 3) and F (rank 6) are relevant; Z is never retrieved.
    assert metrics.recall_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)


def test_recall_with_no_relevant_is_zero() -> None:
    assert metrics.recall_at_k(RANKED, set(), 5) == 0.0


def test_mean_and_safe_div() -> None:
    assert metrics.mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert metrics.mean([]) == 0.0
    assert metrics.safe_div(1, 0) == 0.0
    assert metrics.safe_div(3, 4) == 0.75


def test_binary_counts_and_prf() -> None:
    y_true = [True, True, True, False, False, False, True]
    y_pred = [True, False, True, True, False, False, False]

    counts = metrics.binary_counts(y_true, y_pred)

    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (2, 1, 2, 2)
    p, r, f1 = metrics.precision_recall_f1(y_true, y_pred)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(4 / 7)


def test_prf_without_predictions_or_positives_is_zero_not_nan() -> None:
    assert metrics.precision_recall_f1([True, False], [False, False]) == (0.0, 0.0, 0.0)
    assert metrics.precision_recall_f1([], []) == (0.0, 0.0, 0.0)


def test_perfect_prediction_scores_one() -> None:
    assert metrics.precision_recall_f1([True, False], [True, False]) == (1.0, 1.0, 1.0)


def test_accuracy() -> None:
    assert metrics.accuracy(["a", "b", "c", "a"], ["a", "c", "c", "b"]) == 0.5
    assert metrics.accuracy([], []) == 0.0


def test_macro_f1_hand_computed() -> None:
    # a: P=1, R=1/2 -> F1=2/3; b: P=1/3, R=1 -> F1=1/2; c: F1=0 -> macro = 7/18.
    y_true = ["a", "a", "b", "c"]
    y_pred = ["a", "b", "b", "b"]

    assert metrics.per_class_f1(y_true, y_pred) == pytest.approx({"a": 2 / 3, "b": 0.5, "c": 0.0})
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx(7 / 18)


def test_macro_f1_counts_classes_that_are_only_predicted() -> None:
    # a: F1=2/3; x (never true, predicted once): F1=0 -> macro = 1/3.
    assert metrics.macro_f1(["a", "a"], ["a", "x"]) == pytest.approx(1 / 3)


def test_macro_f1_explicit_labels() -> None:
    assert metrics.macro_f1(["a", "b"], ["a", "b"], labels=["a", "b", "c"]) == pytest.approx(2 / 3)


def test_confusion_matrix_rows_true_columns_predicted() -> None:
    matrix = metrics.confusion_matrix(
        ["a", "a", "b", "c"], ["a", "b", "b", "b"], labels=["a", "b", "c"]
    )

    assert matrix == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]


def test_confusion_matrix_skips_unknown_labels() -> None:
    assert metrics.confusion_matrix(["a", "z"], ["a", "a"], labels=["a"]) == [[1]]


@pytest.mark.parametrize(
    "func",
    [metrics.binary_counts, metrics.accuracy, metrics.macro_f1],
)
def test_length_mismatch_raises(func: object) -> None:
    with pytest.raises(ValueError, match="same length"):
        func([True], [True, False])  # type: ignore[operator]


def test_confusion_matrix_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        metrics.confusion_matrix(["a"], [], labels=["a"])
