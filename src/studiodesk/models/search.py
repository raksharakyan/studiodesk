"""Request and response models for `POST /search`.

Filters accept only enum values and semver strings, so nothing from the request is ever
passed through to Qdrant as a raw filter.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import (
    Component,
    DocType,
    Platform,
    Severity,
    Version,
    version_to_int,
)

QUERY_MAX_CHARS = 500
TOP_K_DEFAULT = 5
TOP_K_MAX = 20
SNIPPET_MAX_CHARS = 500
_ELLIPSIS = "..."


class SearchFilters(BaseModel):
    """Optional metadata filters; every non-empty field must match (AND).

    Within a list, any value may match (OR). The version range is inclusive.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_types: list[DocType] = Field(default_factory=list, max_length=len(DocType))
    platforms: list[Platform] = Field(default_factory=list, max_length=len(Platform))
    severities: list[Severity] = Field(default_factory=list, max_length=len(Severity))
    components: list[Component] = Field(default_factory=list, max_length=len(Component))
    version_min: Version | None = None
    version_max: Version | None = None

    @model_validator(mode="after")
    def _valid_version_range(self) -> Self:
        """Require packable versions and `version_min <= version_max`."""
        low = version_to_int(self.version_min) if self.version_min else None
        high = version_to_int(self.version_max) if self.version_max else None
        if low is not None and high is not None and low > high:
            raise ValueError("version_min must not be greater than version_max")
        return self


class SearchRequest(BaseModel):
    """Body of `POST /search`."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=QUERY_MAX_CHARS)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=TOP_K_MAX)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SearchHitMetadata(BaseModel):
    """Filterable metadata of the chunk behind a hit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_index: int
    platform: list[Platform]
    version: str | None
    severity: Severity | None
    component: Component | None
    created_at: str | None
    duplicate_of: str | None
    related_bug: str | None


class SearchHit(BaseModel):
    """One retrieved chunk with its similarity score.

    `title` and `snippet` are untrusted document text, returned as data.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float
    doc_id: str
    doc_type: DocType
    title: str
    snippet: str = Field(max_length=SNIPPET_MAX_CHARS)
    metadata: SearchHitMetadata

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> "SearchHit":
        """Build a hit from a stored chunk, truncating its text to a snippet."""
        return cls(
            score=score,
            doc_id=chunk.doc_id,
            doc_type=chunk.doc_type,
            title=chunk.title,
            snippet=make_snippet(chunk.text),
            metadata=SearchHitMetadata(
                chunk_index=chunk.chunk_index,
                platform=chunk.platform,
                version=chunk.version,
                severity=chunk.severity,
                component=chunk.component,
                created_at=chunk.created_at,
                duplicate_of=chunk.duplicate_of,
                related_bug=chunk.related_bug,
            ),
        )


class SearchResponse(BaseModel):
    """Body returned by `POST /search`, hits ordered by descending score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hits: list[SearchHit]


def make_snippet(text: str, limit: int = SNIPPET_MAX_CHARS) -> str:
    """Return `text` cut to at most `limit` chars, ending in an ellipsis when truncated."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS
