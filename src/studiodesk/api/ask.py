"""`POST /ask`: answer a question from retrieved documents with verified citations."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter

from studiodesk.agent.answer import answer_question
from studiodesk.agent.retrieval import Retriever
from studiodesk.api.deps import get_llm, get_retriever, get_settings
from studiodesk.api.errors import llm_http_error, vector_store_http_error
from studiodesk.config import Settings
from studiodesk.llm import LLMClient, LLMError
from studiodesk.models.answer import AnswerResponse, AskRequest
from studiodesk.vectorstore import VectorStoreError


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Create the ask router, rate-limited per client IP by `limiter` at `rate_limit`."""
    router = APIRouter(tags=["agent"])

    @router.post("/ask", response_model=AnswerResponse)
    @limiter.limit(rate_limit)
    def ask(
        request: Request,
        body: AskRequest,
        retriever: Annotated[Retriever, Depends(get_retriever)],
        llm: Annotated[LLMClient, Depends(get_llm)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> AnswerResponse:
        """Answer `question` using only retrieved documents; sources are the cited ones."""
        try:
            return answer_question(
                body.question, body.filters, retriever, llm, top_k=settings.answer_top_k
            )
        except VectorStoreError:
            raise vector_store_http_error() from None
        except LLMError as exc:
            raise llm_http_error(exc) from None

    return router
