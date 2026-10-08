"""Advisory duplicate detection: similarity search over bug reports, then one LLM judgement.

Pipeline: embed the new report's title + description, search `doc_type=bug_report` top-k
(`dup_search_k`), keep candidates scoring >= `dup_candidate_threshold`, and ask the LLM to
judge all candidates in one call. Nothing is mutated; the verdict only decides whether an
issue is *proposed* (the user still has to confirm it).

Verdict:
- `duplicate`: the LLM says yes for a candidate whose score is >= `dup_auto_threshold`
  (the highest-scoring such candidate becomes `duplicate_of`).
- `possible_duplicate`: the LLM says yes below the auto threshold, or a candidate scores
  >= the auto threshold although the LLM says no (disagreement is surfaced, not hidden).
- `new`: no candidate, or the LLM says no to every candidate below the auto threshold.

`duplicate_of` is the canonical original: the matched candidate's `duplicate_of` payload
field is followed up to the root (at most 5 hops, cycle-safe), e.g. a paraphrase of
BUG-0002 that best matches BUG-0006 (itself labelled duplicate_of BUG-0002) reports
`duplicate_of=BUG-0002, matched_report=BUG-0006`. Candidates are reported unchanged.

Threshold calibration (pinned all-MiniLM-L6-v2, in-memory Qdrant, whole synthetic dataset,
query = title + description, `doc_types=[bug_report]`, query bug excluded):

- Score of the labelled original for each of the 16 duplicates: min 0.501 (BUG-0011 ->
  BUG-0002), then 0.563, 0.596, 0.602, 0.615, 0.634, 0.655, 0.681, 0.711, 0.711, 0.717,
  0.734, 0.762, 0.775, 0.777, max 0.786. All 16 originals are in the top 5.
- Top-1 score of the hard negatives (look-alikes that are not duplicates): BUG-0012 0.684,
  BUG-0043 0.664, BUG-0021 0.627, BUG-0056 0.547, BUG-0033 0.530, BUG-0066 0.388,
  BUG-0064 0.326.
- Top-1 score of the other non-duplicate bugs: max 0.587 (BUG-0054), then 0.525, 0.496,
  0.491, 0.486; most are below 0.45.

`dup_candidate_threshold = 0.48`: recall first. It keeps every labelled original (lowest
0.501, a 0.02 margin) while dropping most unrelated bugs, so the LLM is not called at all
for clearly new reports. Scores do not separate duplicates from hard negatives
(0.50-0.79 vs. up to 0.68), so the LLM, not the threshold, makes that call.

`dup_auto_threshold = 0.70`: above the highest hard-negative score (0.684), so even if the
LLM wrongly agrees on a look-alike, the verdict is at most `possible_duplicate`. 8 of 16
labelled duplicates reach it; the rest land in `possible_duplicate` when the LLM agrees,
which still shows the candidate but keeps the issue proposal available.
"""

import logging
import math
from collections.abc import Callable, Collection

from studiodesk.agent.retrieval import Retriever
from studiodesk.llm import LLMClient
from studiodesk.models.bugs import (
    REASON_MAX_CHARS,
    CandidateJudgement,
    DuplicateCandidate,
    DuplicateCheck,
    DuplicateJudgements,
    DuplicateVerdict,
    NewBugReport,
)
from studiodesk.models.chunks import Chunk
from studiodesk.prompts import DUPLICATE_SYSTEM_PROMPT, build_duplicate_prompt
from studiodesk.vectorstore import ScoredChunk

logger = logging.getLogger(__name__)

NO_JUDGEMENT_REASON = "No judgement returned for this candidate."
MAX_CANONICAL_DEPTH = 5


def check_duplicates(
    report: NewBugReport,
    retriever: Retriever,
    llm: LLMClient,
    *,
    search_k: int,
    candidate_threshold: float,
    auto_threshold: float,
    exclude_ids: Collection[str] = (),
) -> DuplicateCheck:
    """Return the advisory duplicate verdict for `report`.

    Args:
        report: The submitted report (untrusted).
        retriever: Search over the ingested documents.
        llm: Judges candidates; not called when there are none.
        search_k: Bug reports retrieved before thresholding.
        candidate_threshold: Minimum score to be judged by the LLM.
        auto_threshold: Minimum score for an LLM "yes" to become `duplicate`.
        exclude_ids: Doc ids never considered (e.g. the report's own id in evals).

    Raises:
        VectorStoreError: if retrieval fails.
        LLMError: if the LLM call fails.
    """
    hits = retriever.search_bug_reports(report.search_text, search_k, exclude_ids=exclude_ids)
    shortlisted = [hit for hit in hits if hit.score >= candidate_threshold]
    if not shortlisted:
        return DuplicateCheck(verdict=DuplicateVerdict.NEW, auto_threshold=auto_threshold)
    result = llm.structured(
        DUPLICATE_SYSTEM_PROMPT,
        build_duplicate_prompt(report.prompt_text(), [hit.chunk for hit in shortlisted]),
        DuplicateJudgements,
    )
    candidates = merge_judgements(shortlisted, result.judgements)
    check = decide_verdict(candidates, auto_threshold)
    # Lazy: the store is only queried when a candidate has a `duplicate_of` parent.
    canonical_ids = resolve_canonical_ids(
        shortlisted, lambda doc_id: retriever.get_bug_report(doc_id), exclude_ids
    )
    return check.model_copy(
        update={
            "canonical_ids": canonical_ids,
            "duplicate_of": canonical_ids.get(check.duplicate_of, check.duplicate_of)
            if check.duplicate_of
            else None,
        }
    )


def canonical_original(
    doc_id: str,
    parent: str | None,
    lookup: Callable[[str], Chunk | None],
    *,
    excluded: Collection[str] = (),
    max_depth: int = MAX_CANONICAL_DEPTH,
) -> str:
    """Follow `duplicate_of` links from `doc_id` (whose parent is `parent`) to the root.

    Stops at a report without `duplicate_of`, at a report that is not stored, before an
    `excluded` report (treated as absent from the index), on a cycle, or after
    `max_depth` hops, returning the last id reached.
    """
    current = doc_id
    visited = {doc_id}
    for _ in range(max_depth):
        if parent is None or parent in visited or parent in excluded:
            break
        visited.add(parent)
        current = parent
        chunk = lookup(current)
        parent = chunk.duplicate_of if chunk is not None else None
    return current


def resolve_canonical_ids(
    hits: list[ScoredChunk],
    lookup: Callable[[str], Chunk | None],
    exclude_ids: Collection[str] = (),
) -> dict[str, str]:
    """Map each hit's doc id to its canonical original (lookups cached per call).

    Excluded ids (e.g. the query report itself in evals) are treated as absent, so a chain
    is never resolved to or through them.

    Raises:
        VectorStoreError: if a lookup fails.
    """
    cache: dict[str, Chunk | None] = {}

    def cached(doc_id: str) -> Chunk | None:
        if doc_id not in cache:
            cache[doc_id] = lookup(doc_id)
        return cache[doc_id]

    resolved: dict[str, str] = {}
    for hit in hits:
        resolved[hit.chunk.doc_id] = canonical_original(
            hit.chunk.doc_id, hit.chunk.duplicate_of, cached, excluded=exclude_ids
        )
    return resolved


def clamp_confidence(value: float) -> float:
    """Clamp to [0, 1]; a non-finite value (NaN, +-inf) from the model becomes 0.0."""
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.0


def merge_judgements(
    hits: list[ScoredChunk], judgements: list[CandidateJudgement]
) -> list[DuplicateCandidate]:
    """Attach the LLM judgement to each shortlisted hit, in score order.

    Judgements for ids that were not shortlisted are ignored; the first judgement per id
    wins; a hit without a judgement counts as "not a duplicate". Confidence is clamped to
    [0, 1] (non-finite -> 0.0) and the reason truncated, since the schema cannot enforce
    either.
    """
    by_id: dict[str, CandidateJudgement] = {}
    for item in judgements:
        by_id.setdefault(item.candidate_id, item)
    ignored = len(set(by_id) - {hit.chunk.doc_id for hit in hits})
    if ignored:
        logger.warning("ignored judgements for unknown candidates", extra={"count": ignored})
    candidates: list[DuplicateCandidate] = []
    for hit in hits:
        judgement = by_id.get(hit.chunk.doc_id)
        candidates.append(
            DuplicateCandidate(
                doc_id=hit.chunk.doc_id,
                title=hit.chunk.title,
                score=hit.score,
                is_duplicate=judgement.is_duplicate if judgement else False,
                confidence=clamp_confidence(judgement.confidence) if judgement else 0.0,
                reason=(judgement.reason.strip() if judgement else NO_JUDGEMENT_REASON)[
                    :REASON_MAX_CHARS
                ],
            )
        )
    return candidates


def decide_verdict(candidates: list[DuplicateCandidate], auto_threshold: float) -> DuplicateCheck:
    """Apply the verdict rules from the module docstring to judged candidates."""
    confirmed = [c for c in candidates if c.is_duplicate and c.score >= auto_threshold]
    if confirmed:
        best = max(confirmed, key=lambda c: c.score)
        return DuplicateCheck(
            verdict=DuplicateVerdict.DUPLICATE,
            duplicate_of=best.doc_id,
            matched_report=best.doc_id,
            candidates=candidates,
            auto_threshold=auto_threshold,
        )
    verdict = (
        DuplicateVerdict.POSSIBLE_DUPLICATE
        if any(c.is_duplicate or c.score >= auto_threshold for c in candidates)
        else DuplicateVerdict.NEW
    )
    return DuplicateCheck(verdict=verdict, candidates=candidates, auto_threshold=auto_threshold)
