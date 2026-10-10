"""The eval case files parse, have the planned shape, and every label is checkable.

These tests guard the *labels*, not the scores: doc ids must exist, key facts must be
stated in the expected documents, calibration cases must mirror the dataset, and held-out
text must not copy phrases from the corpus.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest
from evals.cases import CASE_FILES, CASES_DIR, CaseSet, load_cases
from evals.judge import fact_present
from evals.suites import routing_cases

from studiodesk.data.loader import Dataset
from studiodesk.ingest.chunking import chunk_dataset
from studiodesk.models.documents import Platform

INJECTED_DOCS = {"BUG-0014", "BUG-0026", "BUG-0041", "BUG-0054", "DOC-troubleshooting-guide"}
HARD_NEGATIVES = {
    "BUG-0012",
    "BUG-0043",
    "BUG-0021",
    "BUG-0056",
    "BUG-0033",
    "BUG-0066",
    "BUG-0064",
}
NGRAM = 6


@pytest.fixture(scope="module")
def cases() -> CaseSet:
    return load_cases()


@pytest.fixture(scope="module")
def doc_texts(dataset: Dataset) -> dict[str, str]:
    """Full text per doc id (all chunks joined)."""
    texts: dict[str, list[str]] = {}
    for chunk in chunk_dataset(dataset):
        texts.setdefault(chunk.doc_id, []).append(f"{chunk.title}\n{chunk.text}")
    return {doc_id: "\n".join(parts) for doc_id, parts in texts.items()}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*", text.lower())


# ------------------------------------------------------------------------ parse + shape


def test_every_case_file_parses_line_by_line() -> None:
    for name in CASE_FILES.values():
        for number, line in enumerate((CASES_DIR / name).read_text().splitlines(), 1):
            row = json.loads(line)
            assert {"id", "suite", "split", "inputs", "labels"} <= set(row), f"{name}:{number}"
            assert row["split"] in {"calibration", "heldout"}


def test_case_counts_per_suite_and_split(cases: CaseSet) -> None:
    assert Counter(c.split for c in cases.retrieval) == {"heldout": 25}
    assert Counter(c.labels.expect_insufficient for c in cases.answer) == {False: 15, True: 5}
    assert Counter((c.split, c.labels.kind) for c in cases.duplicates) == {
        ("calibration", "labelled_duplicate"): 16,
        ("calibration", "hard_negative"): 7,
        ("heldout", "paraphrase"): 12,
        ("heldout", "novel"): 10,
    }
    assert Counter(c.labels.kind for c in cases.injection) == {
        "retrieved_injection": 6,
        "duplicate_candidate_injection": 4,
        "user_report_injection": 1,
        "direct_user_injection": 3,
    }


def test_routing_cases_are_leave_one_out_plus_novel(cases: CaseSet, dataset: Dataset) -> None:
    routed = routing_cases(dataset, cases.duplicates)

    loo = [c for c in routed if c.group == "dataset_loo"]
    assert len(loo) == len(dataset.bug_reports) == 66
    assert all(c.exclude_ids == (c.id.removeprefix("route-loo-"),) for c in loo)
    assert sum(c.group == "novel" for c in routed) == 10


def test_case_ids_are_unique(cases: CaseSet) -> None:
    ids = [
        c.id
        for group in (cases.retrieval, cases.answer, cases.duplicates, cases.injection)
        for c in group
    ]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------- referenced ids


def test_every_referenced_doc_id_exists(cases: CaseSet, doc_texts: dict[str, str]) -> None:
    referenced: set[str] = set()
    for c in cases.retrieval:
        referenced |= set(c.labels.relevant_ids)
    for a in cases.answer:
        referenced |= set(a.labels.expected_ids)
    for d in cases.duplicates:
        referenced |= set(d.inputs.exclude_ids)
        referenced |= {x for x in (d.labels.original, d.labels.source_bug) if x}
    for i in cases.injection:
        labels = i.labels
        referenced |= set(labels.forbidden_duplicate_of)
        referenced |= {
            x for x in (labels.injected_id, labels.expect_retrieved, labels.expect_candidate) if x
        }

    assert referenced - set(doc_texts) == set()


def test_forbidden_ids_really_do_not_exist(cases: CaseSet, doc_texts: dict[str, str]) -> None:
    forbidden = {x for i in cases.injection for x in i.labels.forbidden_ids}
    assert forbidden == {"BUG-9999"}
    assert forbidden.isdisjoint(doc_texts)


def test_retrieval_cases_cover_doc_types_platforms_and_versions(cases: CaseSet) -> None:
    relevant = {x for c in cases.retrieval for x in c.labels.relevant_ids}
    prefixes = {doc_id.split("-")[0] for doc_id in relevant}
    assert prefixes == {"BUG", "CRASH", "PATCH", "DOC"}
    assert {c.labels.platform for c in cases.retrieval} >= set(Platform)
    assert len({c.labels.version for c in cases.retrieval if c.labels.version}) >= 6
    assert {c.labels.voice for c in cases.retrieval} == {"player", "dev", "support"}


# ------------------------------------------------------------------------- answer facts


def test_key_facts_are_stated_in_an_expected_doc(cases: CaseSet, doc_texts: dict[str, str]) -> None:
    for case in cases.answer:
        source = "\n".join(doc_texts[d] for d in case.labels.expected_ids)
        for alternatives in case.labels.key_facts:
            assert fact_present(source, alternatives), f"{case.id}: {alternatives} not in docs"


def test_answerable_cases_have_one_to_three_facts(cases: CaseSet) -> None:
    for case in cases.answer:
        n = len(case.labels.key_facts)
        assert (n == 0) if case.labels.expect_insufficient else (1 <= n <= 3), case.id


def test_unanswerable_topics_are_absent_from_the_corpus(
    cases: CaseSet, doc_texts: dict[str, str]
) -> None:
    corpus = " ".join(doc_texts.values()).lower()
    for absent in (
        "pvp",
        "season pass",
        "virtual reality",
        " vr ",
        "quest 3",
        "dlss",
        "ray-traced",
        "ray tracing",
    ):
        assert absent not in corpus, absent
    assert all(c.labels.expected_ids == [] for c in cases.answer if c.labels.expect_insufficient)


# ------------------------------------------------------------------------- duplicates


def test_calibration_cases_mirror_the_dataset(cases: CaseSet, dataset: Dataset) -> None:
    bugs = {b.id: b for b in dataset.bug_reports}
    calibration = [c for c in cases.duplicates if c.split == "calibration"]
    labelled = {b.id for b in dataset.bug_reports if b.duplicate_of}

    assert {
        c.labels.source_bug for c in calibration if c.labels.kind == "labelled_duplicate"
    } == labelled
    assert {
        c.labels.source_bug for c in calibration if c.labels.kind == "hard_negative"
    } == HARD_NEGATIVES
    for case in calibration:
        bug = bugs[case.labels.source_bug or ""]
        assert case.inputs.title == bug.title
        assert case.inputs.description == bug.description
        assert case.inputs.platform == bug.platform
        assert case.inputs.version == bug.version
        assert case.inputs.exclude_ids == [bug.id], "the query bug must be excluded"
        assert case.labels.original == bug.duplicate_of


def test_heldout_paraphrases_target_bugs_outside_the_duplicate_pairs(
    cases: CaseSet, dataset: Dataset
) -> None:
    paired = {b.id for b in dataset.bug_reports if b.duplicate_of}
    paired |= {b.duplicate_of for b in dataset.bug_reports if b.duplicate_of}
    originals = [c.labels.original for c in cases.duplicates if c.labels.kind == "paraphrase"]

    assert len(set(originals)) == 12
    assert set(originals).isdisjoint(paired | HARD_NEGATIVES | INJECTED_DOCS)
    for case in cases.duplicates:
        if case.split == "heldout":
            assert case.inputs.exclude_ids == []


def test_novel_bugs_are_labelled_for_routing(cases: CaseSet) -> None:
    novel = [c for c in cases.duplicates if c.labels.kind == "novel"]
    assert all(
        c.labels.expected == "new" and c.labels.component and c.labels.severity for c in novel
    )
    assert len({c.labels.component for c in novel}) == 10


def test_heldout_text_does_not_copy_corpus_phrases(
    cases: CaseSet, doc_texts: dict[str, str]
) -> None:
    corpus_words = _words(" ".join(doc_texts.values()))
    grams = {tuple(corpus_words[i : i + NGRAM]) for i in range(len(corpus_words) - NGRAM + 1)}
    titles = {" ".join(_words(t.split("\n", 1)[0])) for t in doc_texts.values()}
    texts = [(c.id, c.inputs.question) for c in (*cases.retrieval, *cases.answer)]
    texts += [
        (c.id, f"{c.inputs.title} {c.inputs.description}")
        for c in cases.duplicates
        if c.split == "heldout"
    ]
    for case_id, text in texts:
        words = _words(text)
        copied = [
            " ".join(words[i : i + NGRAM])
            for i in range(len(words) - NGRAM + 1)
            if tuple(words[i : i + NGRAM]) in grams
        ]
        assert not copied, f"{case_id} copies corpus phrases: {copied}"
        assert " ".join(words) not in titles, case_id


# -------------------------------------------------------------------------- injection


def test_injection_cases_cover_every_injected_doc(
    cases: CaseSet, doc_texts: dict[str, str]
) -> None:
    retrieved = {
        c.labels.expect_retrieved for c in cases.injection if c.labels.kind == "retrieved_injection"
    }
    candidates = {c.labels.expect_candidate for c in cases.injection if c.labels.expect_candidate}

    assert retrieved == INJECTED_DOCS
    assert candidates == INJECTED_DOCS - {"DOC-troubleshooting-guide"}
    for case in cases.injection:
        marker = case.labels.expect_chunk_contains
        if marker:
            assert marker in doc_texts[case.labels.expect_retrieved or ""]


def test_direct_injection_questions_are_hostile(cases: CaseSet) -> None:
    direct = [
        c.inputs.question or "" for c in cases.injection if c.labels.kind == "direct_user_injection"
    ]
    joined = " ".join(direct).lower()
    assert "system prompt" in joined
    assert "environment variables" in joined
    assert "bug-9999" in joined


# ---------------------------------------------------------------------- invalid input


def _write_cases(directory: Path, **overrides: str) -> None:
    for suite, name in CASE_FILES.items():
        (directory / name).write_text(overrides.get(suite, (CASES_DIR / name).read_text()))


def test_invalid_json_names_file_and_line(tmp_path: Path) -> None:
    good = (CASES_DIR / "answer.jsonl").read_text()
    _write_cases(tmp_path, answer=good + "{not json\n")

    with pytest.raises(ValueError, match=r"answer.jsonl:21"):
        load_cases(tmp_path)


@pytest.mark.parametrize(
    "line",
    [
        # unknown field
        '{"id": "x", "suite": "answer", "split": "heldout", "inputs": {"question": "q"}, '
        '"labels": {"expect_insufficient": true}, "extra": 1}',
        # answerable without facts
        '{"id": "x", "suite": "answer", "split": "heldout", "inputs": {"question": "q"}, '
        '"labels": {"expect_insufficient": false, "expected_ids": ["BUG-0001"]}}',
        # unanswerable with expected ids
        '{"id": "x", "suite": "answer", "split": "heldout", "inputs": {"question": "q"}, '
        '"labels": {"expect_insufficient": true, "expected_ids": ["BUG-0001"]}}',
        # bad split
        '{"id": "x", "suite": "answer", "split": "train", "inputs": {"question": "q"}, '
        '"labels": {"expect_insufficient": true}}',
        # four key facts
        '{"id": "x", "suite": "answer", "split": "heldout", "inputs": {"question": "q"}, '
        '"labels": {"expect_insufficient": false, "expected_ids": ["BUG-0001"], '
        '"key_facts": [["a"], ["b"], ["c"], ["d"]]}}',
    ],
)
def test_invalid_answer_cases_are_rejected(tmp_path: Path, line: str) -> None:
    _write_cases(tmp_path, answer=line + "\n")

    with pytest.raises(ValueError, match="invalid case"):
        load_cases(tmp_path)


def test_duplicate_without_original_is_rejected(tmp_path: Path) -> None:
    line = (
        '{"id": "x", "suite": "duplicates", "split": "heldout", "inputs": {"title": "Some title",'
        ' "description": "d", "platform": "pc", "version": "1.0.0"},'
        ' "labels": {"kind": "paraphrase", "expected": "duplicate"}}'
    )
    _write_cases(tmp_path, duplicates=line + "\n")

    with pytest.raises(ValueError, match="invalid case"):
        load_cases(tmp_path)


def test_repeated_case_ids_are_rejected(tmp_path: Path) -> None:
    first = (CASES_DIR / "answer.jsonl").read_text().splitlines()[0]
    _write_cases(tmp_path, answer=f"{first}\n{first}\n")

    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(tmp_path)


def test_fingerprints_change_with_content(tmp_path: Path) -> None:
    _write_cases(tmp_path)
    before = load_cases(tmp_path).fingerprints
    (tmp_path / "answer.jsonl").write_text((CASES_DIR / "answer.jsonl").read_text() + "\n")

    assert load_cases(tmp_path).fingerprints["answer.jsonl"] != before["answer.jsonl"]
    assert load_cases(tmp_path).fingerprints["retrieval.jsonl"] == before["retrieval.jsonl"]
