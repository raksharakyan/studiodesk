"""Answer a question from retrieved documents, with server-verified citations."""

import logging

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
    retrieved set) are dropped, so every returned source really backs the answer context.

    Raises:
        VectorStoreError: if retrieval fails.
        LLMError: if the LLM call fails.
    """
    hits = retriever.search(question, filters, top_k)
    if not hits:
        return AnswerResponse(answer=NO_CONTEXT_ANSWER, insufficient_context=True, sources=[])
    result = llm.structured(
        ANSWER_SYSTEM_PROMPT,
        build_answer_prompt(question, [hit.chunk for hit in hits]),
        LLMAnswer,
    )
    sources = cited_sources(result.cited_ids, hits)
    dropped = len(set(result.cited_ids)) - len(sources)
    if dropped:
        logger.warning("dropped citations not in the retrieved set", extra={"count": dropped})
    return AnswerResponse(
        answer=_truncate(result.answer.strip(), ANSWER_MAX_CHARS),
        insufficient_context=result.insufficient_context,
        sources=sources,
    )


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
