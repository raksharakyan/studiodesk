"""Advisory duplicate detection: thresholds, LLM yes/no x score matrix, merging rules."""

import math

import pytest
from pydantic import ValidationError

from fakes import FakeLLM, FakeRetriever, judge_all, prompt_doc_ids, scored
from studiodesk.agent.duplicates import (
    NO_JUDGEMENT_REASON,
    check_duplicates,
    decide_verdict,
    merge_judgements,
)
from studiodesk.agent.retrieval import Retriever
from studiodesk.config import Settings
from studiodesk.data.loader import Dataset
from studiodesk.embeddings import Embedder
from studiodesk.models.bugs import (
    REASON_MAX_CHARS,
    CandidateJudgement,
    DuplicateJudgements,
    DuplicateVerdict,
    NewBugReport,
)
from studiodesk.models.documents import DocType
from studiodesk.models.search import SearchFilters
from studiodesk.prompts import DUPLICATE_SYSTEM_PROMPT
from studiodesk.vectorstore import QdrantStore, ScoredChunk

CANDIDATE = 0.5
AUTO = 0.7
REPORT = NewBugReport(
    title="Game crashes on load",
    description="Loading a save crashes the game.",
    platform="pc",
    version="1.0.2",
)


def _judgement(doc_id: str, yes: bool, **kw: object) -> CandidateJudgement:
    fields: dict[str, object] = {"confidence": 0.8, "reason": "same defect"}
    fields.update(kw)
    return CandidateJudgement(candidate_id=doc_id, is_duplicate=yes, **fields)


def _run(
    hits: list[ScoredChunk], judgements: list[CandidateJudgement] | None = None
) -> tuple[object, FakeLLM]:
    llm = FakeLLM({DuplicateJudgements: DuplicateJudgements(judgements=judgements or [])})
    result = check_duplicates(
        REPORT,
        FakeRetriever(hits=hits),
        llm,
        search_k=5,
        candidate_threshold=CANDIDATE,
        auto_threshold=AUTO,
    )
    return result, llm


@pytest.mark.parametrize(
    ("score", "llm_yes", "verdict"),
    [
        (0.85, True, DuplicateVerdict.DUPLICATE),
        (0.70, True, DuplicateVerdict.DUPLICATE),  # auto threshold is inclusive
        (0.85, False, DuplicateVerdict.POSSIBLE_DUPLICATE),  # disagreement is surfaced
        (0.70, False, DuplicateVerdict.POSSIBLE_DUPLICATE),
        (0.69, True, DuplicateVerdict.POSSIBLE_DUPLICATE),
        (0.50, True, DuplicateVerdict.POSSIBLE_DUPLICATE),  # candidate threshold inclusive
        (0.69, False, DuplicateVerdict.NEW),
        (0.50, False, DuplicateVerdict.NEW),
    ],
)
def test_verdict_matrix(score: float, llm_yes: bool, verdict: DuplicateVerdict) -> None:
    result, llm = _run([scored("BUG-0002", score)], [_judgement("BUG-0002", llm_yes)])

    assert result.verdict is verdict
    expected_of = "BUG-0002" if verdict is DuplicateVerdict.DUPLICATE else None
    assert result.duplicate_of == expected_of
    assert result.auto_threshold == AUTO
    assert len(llm.calls) == 1


def test_below_candidate_threshold_is_new_without_llm_call() -> None:
    result, llm = _run([scored("BUG-0002", 0.49), scored("BUG-0003", 0.1)])

    assert result.verdict is DuplicateVerdict.NEW
    assert result.candidates == []
    assert llm.calls == []


def test_no_hits_is_new_without_llm_call() -> None:
    result, llm = _run([])
    assert result.verdict is DuplicateVerdict.NEW
    assert llm.calls == []


def test_only_shortlisted_candidates_reach_the_prompt() -> None:
    hits = [scored("BUG-0002", 0.8), scored("BUG-0003", 0.6), scored("BUG-0004", 0.3)]
    result, llm = _run(hits, [])

    [call] = llm.calls
    assert call.system == DUPLICATE_SYSTEM_PROMPT
    assert prompt_doc_ids(call.user_content) == ["BUG-0002", "BUG-0003"]
    assert "<new_bug_report>" in call.user_content
    assert [c.doc_id for c in result.candidates] == ["BUG-0002", "BUG-0003"]


def test_highest_scoring_confirmed_candidate_wins() -> None:
    hits = [scored("BUG-0005", 0.95), scored("BUG-0002", 0.85), scored("BUG-0003", 0.75)]
    result, _ = _run(
        hits,
        [
            _judgement("BUG-0005", False),
            _judgement("BUG-0003", True),
            _judgement("BUG-0002", True),
        ],
    )

    assert result.verdict is DuplicateVerdict.DUPLICATE
    assert result.duplicate_of == "BUG-0002"


def test_missing_judgement_counts_as_no() -> None:
    result, _ = _run([scored("BUG-0002", 0.6)], [])

    [candidate] = result.candidates
    assert candidate.is_duplicate is False
    assert candidate.confidence == 0.0
    assert candidate.reason == NO_JUDGEMENT_REASON
    assert result.verdict is DuplicateVerdict.NEW


def test_judgements_for_non_shortlisted_ids_are_ignored() -> None:
    hits = [scored("BUG-0002", 0.6), scored("BUG-0009", 0.2)]
    result, _ = _run(
        hits,
        [_judgement("BUG-0009", True), _judgement("BUG-0777", True), _judgement("BUG-0002", False)],
    )

    assert result.verdict is DuplicateVerdict.NEW
    assert result.duplicate_of is None
    assert [c.doc_id for c in result.candidates] == ["BUG-0002"]


def test_first_judgement_per_id_wins() -> None:
    result, _ = _run(
        [scored("BUG-0002", 0.9)], [_judgement("BUG-0002", False), _judgement("BUG-0002", True)]
    )
    assert result.candidates[0].is_duplicate is False


@pytest.mark.parametrize(("raw", "clamped"), [(5.0, 1.0), (-2.0, 0.0), (0.42, 0.42)])
def test_confidence_is_clamped(raw: float, clamped: float) -> None:
    result, _ = _run([scored("BUG-0002", 0.6)], [_judgement("BUG-0002", True, confidence=raw)])
    assert result.candidates[0].confidence == clamped


@pytest.mark.xfail(
    strict=True,
    raises=ValidationError,
    reason="DEFECT: NaN confidence from the LLM passes min/max clamping and fails "
    "DuplicateCandidate validation (uncaught -> 500); agent/duplicates.py:131",
)
def test_nan_confidence_is_handled() -> None:
    judgements = DuplicateJudgements.model_validate_json(
        '{"judgements": [{"candidate_id": "BUG-0002", "is_duplicate": true, '
        '"confidence": NaN, "reason": "r"}]}'
    )
    candidates = merge_judgements([scored("BUG-0002", 0.6)], judgements.judgements)
    assert not math.isnan(candidates[0].confidence)


def test_reason_is_stripped_and_truncated() -> None:
    long_reason = "  " + "r" * (REASON_MAX_CHARS + 100) + "  "
    result, _ = _run([scored("BUG-0002", 0.6)], [_judgement("BUG-0002", True, reason=long_reason)])
    assert result.candidates[0].reason == "r" * REASON_MAX_CHARS


def test_decide_verdict_with_no_candidates_is_new() -> None:
    assert decide_verdict([], 0.7).verdict is DuplicateVerdict.NEW


def test_settings_reject_candidate_above_auto() -> None:
    with pytest.raises(ValidationError, match="dup_candidate_threshold"):
        Settings(_env_file=None, dup_candidate_threshold=0.8, dup_auto_threshold=0.7)
    Settings(_env_file=None, dup_candidate_threshold=0.7, dup_auto_threshold=0.7)


# --------------------------------------------------------------------------- real retriever


class SpyStore:
    """Wraps a QdrantStore and records the filters of every search."""

    def __init__(self, inner: QdrantStore) -> None:
        self.inner = inner
        self.filters: list[SearchFilters | None] = []

    def search(self, vector: list[float], filters: SearchFilters | None, top_k: int) -> list:
        self.filters.append(filters)
        return self.inner.search(vector, filters, top_k)


def test_only_bug_reports_are_searched_and_self_is_excluded(
    ingested_store: QdrantStore, fake_embedder: Embedder, dataset: Dataset
) -> None:
    bug = next(b for b in dataset.bug_reports if b.id == "BUG-0002")
    report = NewBugReport(
        title=bug.title, description=bug.description, platform=bug.platform, version=bug.version
    )
    spy = SpyStore(ingested_store)
    llm = FakeLLM({DuplicateJudgements: judge_all(True)})

    result = check_duplicates(
        report,
        Retriever(fake_embedder, spy),
        llm,
        search_k=5,
        candidate_threshold=0.0,
        auto_threshold=1.0,
        exclude_ids={"BUG-0002"},
    )

    assert spy.filters and all(
        f is not None and f.doc_types == [DocType.BUG_REPORT] for f in spy.filters
    )
    ids = [c.doc_id for c in result.candidates]
    assert len(ids) == 5  # over-fetch keeps k results despite the exclusion
    assert "BUG-0002" not in ids
    assert all(i.startswith("BUG-") for i in ids)
    assert prompt_doc_ids(llm.calls[0].user_content) == ids


def test_without_exclusion_identical_report_is_top_candidate(
    ingested_store: QdrantStore, fake_embedder: Embedder, dataset: Dataset
) -> None:
    bug = next(b for b in dataset.bug_reports if b.id == "BUG-0002")
    report = NewBugReport(
        title=bug.title, description=bug.description, platform=bug.platform, version=bug.version
    )
    llm = FakeLLM({DuplicateJudgements: judge_all(True)})

    result = check_duplicates(
        report,
        Retriever(fake_embedder, ingested_store),
        llm,
        search_k=5,
        candidate_threshold=0.0,
        auto_threshold=0.5,
    )

    assert result.candidates[0].doc_id == "BUG-0002"
    assert result.verdict is DuplicateVerdict.DUPLICATE
    assert result.duplicate_of == "BUG-0002"
