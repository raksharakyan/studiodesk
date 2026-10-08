"""Models for `POST /ask`, plus the schema the LLM must answer in.

`LLMAnswer` is deliberately unconstrained (no length or pattern limits): structured outputs
only carry such limits as descriptions, so the server enforces them after parsing instead
(see `studiodesk.agent.answer`).
"""

from pydantic import BaseModel, ConfigDict, Field

from studiodesk.models.documents import DocType
from studiodesk.models.search import SNIPPET_MAX_CHARS, SearchFilters

QUESTION_MAX_CHARS = 1_000
ANSWER_MAX_CHARS = 4_000
MAX_CITED_IDS = 20


class AskRequest(BaseModel):
    """Body of `POST /ask`."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=QUESTION_MAX_CHARS)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class LLMAnswer(BaseModel):
    """Structured output requested from the LLM for a question."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Plain-text answer, at most about 150 words.")
    cited_ids: list[str] = Field(description="Ids of the documents the answer relies on.")
    insufficient_context: bool = Field(
        description="True if the documents do not contain enough information to answer."
    )


class AnswerSource(BaseModel):
    """A retrieved document the answer cites. `title`/`snippet` are untrusted data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    doc_type: DocType
    title: str
    snippet: str = Field(max_length=SNIPPET_MAX_CHARS)
    score: float


class AnswerResponse(BaseModel):
    """Body returned by `POST /ask`. Every source was actually retrieved for this request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(max_length=ANSWER_MAX_CHARS)
    insufficient_context: bool
    sources: list[AnswerSource] = Field(max_length=MAX_CITED_IDS)
    # Distinct doc ids cited by the model (list or bracketed in the text) but not retrieved.
    removed_citations: int = Field(ge=0)
