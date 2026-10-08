"""answer_question: server-verified citations, no-retrieval path and error propagation."""

import pytest

from fakes import FakeLLM, FakeRetriever, scored
from studiodesk.agent.answer import (
    NO_CONTEXT_ANSWER,
    UNVERIFIED_SOURCE,
    answer_question,
    strip_unverified_citations,
)
from studiodesk.llm import LLMInvalidOutput, LLMUnavailable
from studiodesk.models.answer import ANSWER_MAX_CHARS, MAX_CITED_IDS, AnswerResponse, LLMAnswer
from studiodesk.models.documents import DocType
from studiodesk.prompts import ANSWER_SYSTEM_PROMPT


def _hits() -> FakeRetriever:
    return FakeRetriever(
        hits=[
            scored("BUG-0001", 0.9, title="Save corrupted"),
            scored("PATCH-1.0.1", 0.8, title="Patch 1.0.1", doc_type=DocType.PATCH_NOTE),
            scored("BUG-0001", 0.7, title="Save corrupted (2nd chunk)"),
        ]
    )


def _ask(
    answer: LLMAnswer, retriever: FakeRetriever | None = None, top_k: int = 6
) -> tuple[AnswerResponse, FakeLLM]:
    llm = FakeLLM({LLMAnswer: answer})
    result = answer_question("why?", None, retriever or _hits(), llm, top_k=top_k)
    return result, llm


def test_retrieved_citations_are_kept_in_order_once_with_best_chunk() -> None:
    result, _ = _ask(
        LLMAnswer(
            answer="Fixed in 1.0.1 [PATCH-1.0.1], see [BUG-0001].",
            cited_ids=["PATCH-1.0.1", "BUG-0001", "BUG-0001"],
            insufficient_context=False,
        )
    )

    assert result.answer == "Fixed in 1.0.1 [PATCH-1.0.1], see [BUG-0001]."
    assert [s.doc_id for s in result.sources] == ["PATCH-1.0.1", "BUG-0001"]
    assert result.sources[1].score == 0.9
    assert result.sources[1].title == "Save corrupted"
    assert result.removed_citations == 0


def test_cited_ids_not_retrieved_are_dropped_and_counted() -> None:
    result, _ = _ask(
        LLMAnswer(
            answer="See the notes.",
            cited_ids=["BUG-0001", "BUG-9999", "CRASH-0042", "BUG-9999"],
            insufficient_context=False,
        )
    )

    assert [s.doc_id for s in result.sources] == ["BUG-0001"]
    assert result.removed_citations == 2


def test_bracketed_unretrieved_ids_are_replaced_and_counted_once_per_id() -> None:
    result, _ = _ask(
        LLMAnswer(
            answer="[BUG-0001] ok, [BUG-9999] and [BUG-9999] and [DOC-made-up] and [CRASH-0007]",
            cited_ids=["BUG-0001", "BUG-9999"],
            insufficient_context=False,
        )
    )

    assert result.answer == (
        f"[BUG-0001] ok, {UNVERIFIED_SOURCE} and {UNVERIFIED_SOURCE} and "
        f"{UNVERIFIED_SOURCE} and {UNVERIFIED_SOURCE}"
    )
    # BUG-9999 (list + text), DOC-made-up, CRASH-0007 -> 3 distinct ids.
    assert result.removed_citations == 3


def test_strip_unverified_ignores_non_id_brackets() -> None:
    text = "[note] [BUG-12] [bug-0001] [PATCH-1.0] [BUG-0001]"
    cleaned, removed = strip_unverified_citations(text, {"BUG-0001"})
    assert cleaned == text
    assert removed == set()


def test_no_retrieval_returns_insufficient_context_without_calling_llm() -> None:
    result, llm = _ask(
        LLMAnswer(answer="x", cited_ids=[], insufficient_context=False), FakeRetriever(hits=[])
    )

    assert llm.calls == []
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.insufficient_context is True
    assert result.sources == []


def test_insufficient_context_flag_is_passed_through() -> None:
    result, _ = _ask(LLMAnswer(answer="Not enough info.", cited_ids=[], insufficient_context=True))
    assert result.insufficient_context is True


def test_llm_receives_frozen_system_prompt_and_delimited_documents() -> None:
    _, llm = _ask(LLMAnswer(answer="a", cited_ids=[], insufficient_context=False))
    _, llm2 = _ask(LLMAnswer(answer="b", cited_ids=[], insufficient_context=False))

    assert llm.calls[0].system == ANSWER_SYSTEM_PROMPT == llm2.calls[0].system
    assert llm.calls[0].user_content.count("<document ") == 3
    assert llm.calls[0].schema is LLMAnswer


def test_long_answers_are_truncated() -> None:
    result, _ = _ask(LLMAnswer(answer="x" * 10_000, cited_ids=[], insufficient_context=False))
    assert len(result.answer) == ANSWER_MAX_CHARS
    assert result.answer.endswith("...")


def test_sources_are_capped() -> None:
    hits = [scored(f"BUG-{i:04d}", 0.5) for i in range(1, 30)]
    result, _ = _ask(
        LLMAnswer(answer="a", cited_ids=[h.chunk.doc_id for h in hits], insufficient_context=False),
        FakeRetriever(hits=hits),
        top_k=30,
    )
    assert len(result.sources) == MAX_CITED_IDS


@pytest.mark.parametrize("error", [LLMUnavailable("down"), LLMInvalidOutput("bad")])
def test_llm_errors_propagate(error: Exception) -> None:
    llm = FakeLLM({LLMAnswer: error})
    with pytest.raises(type(error)):
        answer_question("q", None, _hits(), llm, top_k=6)
