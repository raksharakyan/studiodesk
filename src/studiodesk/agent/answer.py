"""Answer a question from retrieved documents, with server-verified citations."""

import logging
import re

from studiodesk.agent.retrieval import Retriever
from studiodesk.llm import LLMClient
from studiodesk.models.answer import (
    ANSWER_MAX_CHARS,
    MAX_CITED_IDS,
    AnswerResponse,
    AnswerSource,
    LLMAnswer,
)
from studiodesk.models.search import SearchFilters, make_snippet
from studiodesk.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from studiodesk.vectorstore import ScoredChunk

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "I could not find any bug reports, crash logs, patch notes or support docs about that."
)
_ELLIPSIS = "..."
UNVERIFIED_SOURCE = "[unverified source]"
DOC_ID_PATTERN = r"(?:BUG-\d{4}|CRASH-\d{4}|PATCH-\d+\.\d+\.\d+|DOC-[a-z0-9]+(?:-[a-z0-9]+)*)"
# Boundaries: not part of a longer token on either side (letters, digits, `_`, `-`, or a
# `.` that continues a version, e.g. PATCH-1.0.1.5); a sentence-final `.` is fine.
_ID_START = r"(?<![A-Za-z0-9_.\-])"
_ID_END = r"(?![A-Za-z0-9_\-]|\.\d)"
# A lone bracketed id ("[BUG-0001]") is replaced as a whole so no double brackets appear;
# any other occurrence (bare, in lists, in parentheses) is replaced token by token.
DOC_ID_RE = re.compile(
    rf"\[\s*(?P<lone>{DOC_ID_PATTERN})\s*\]|{_ID_START}(?P<bare>{DOC_ID_PATTERN}){_ID_END}"
)


def answer_question(
    question: str,
    filters: SearchFilters | None,
    retriever: Retriever,
    llm: LLMClient,
    *,
    top_k: int,
) -> AnswerResponse:
    """Retrieve `top_k` chunks, ask the LLM, and keep only citations that were retrieved.

    With nothing retrieved the LLM is not called. Cited ids the model invents (not in the
    retrieved set) are dropped, and any doc id in the answer text that was not retrieved
    becomes `[unverified source]`; `removed_citations` counts the distinct ids
    removed either way. This does not rely on the model following instructions.

    Raises:
        VectorStoreError: if retrieval fails.
        LLMError: if the LLM call fails.
    """
    hits = retriever.search(question, filters, top_k)
    if not hits:
        return AnswerResponse(
            answer=NO_CONTEXT_ANSWER, insufficient_context=True, sources=[], removed_citations=0
        )
    result = llm.structured(
        ANSWER_SYSTEM_PROMPT,
        build_answer_prompt(question, [hit.chunk for hit in hits]),
        LLMAnswer,
    )
    retrieved_ids = {hit.chunk.doc_id for hit in hits}
    sources = cited_sources(result.cited_ids, hits)
    answer, text_removed = strip_unverified_citations(result.answer, retrieved_ids)
    removed = text_removed | (set(result.cited_ids) - retrieved_ids)
    if removed:
        logger.warning("removed citations not in the retrieved set", extra={"count": len(removed)})
    return AnswerResponse(
        answer=_truncate(answer.strip(), ANSWER_MAX_CHARS),
        insufficient_context=result.insufficient_context,
        sources=sources,
        removed_citations=len(removed),
    )


def strip_unverified_citations(text: str, retrieved_ids: set[str]) -> tuple[str, set[str]]:
    """Replace every doc id in `text` that was not retrieved with `[unverified source]`.

    Ids are found anywhere (bracketed, in lists like `[BUG-0001, BUG-9999]`, in
    parentheses or bare); retrieved ids are kept. Returns the cleaned text and the
    distinct ids that were replaced.
    """
    removed: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        doc_id = match.group("lone") or match.group("bare")
        if doc_id in retrieved_ids:
            return match.group(0)
        removed.add(doc_id)
        return UNVERIFIED_SOURCE

    return DOC_ID_RE.sub(replace, text), removed


def cited_sources(cited_ids: list[str], hits: list[ScoredChunk]) -> list[AnswerSource]:
    """One source per cited id that was retrieved, in citation order (best chunk per doc)."""
    best: dict[str, ScoredChunk] = {}
    for hit in hits:
        current = best.get(hit.chunk.doc_id)
        if current is None or hit.score > current.score:
            best[hit.chunk.doc_id] = hit
    sources: list[AnswerSource] = []
    for doc_id in dict.fromkeys(cited_ids):
        cited = best.get(doc_id)
        if cited is None:
            continue
        sources.append(
            AnswerSource(
                doc_id=cited.chunk.doc_id,
                doc_type=cited.chunk.doc_type,
                title=cited.chunk.title,
                snippet=make_snippet(cited.chunk.text),
                score=cited.score,
            )
        )
        if len(sources) == MAX_CITED_IDS:
            break
    return sources


def _truncate(text: str, limit: int) -> str:
    """Cut `text` to at most `limit` chars, marking the cut with an ellipsis."""
    return text if len(text) <= limit else text[: limit - len(_ELLIPSIS)] + _ELLIPSIS
