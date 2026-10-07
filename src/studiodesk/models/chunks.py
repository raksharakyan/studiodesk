"""The `Chunk` model: one embeddable piece of a document plus its filterable metadata.

A chunk's fields are stored verbatim as the Qdrant point payload. Its text is untrusted
data: it is stored, embedded and returned, never interpreted.
"""

import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from studiodesk.models.documents import (
    BUG_ID_PATTERN,
    Component,
    DocType,
    Platform,
    Severity,
    Version,
)

# Fixed namespace so a chunk's point id is a pure function of `doc_id#chunk_index`.
CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/raksharakyan/studiodesk")

OptionalBugId = Annotated[str | None, Field(pattern=BUG_ID_PATTERN)]


def chunk_point_id(doc_id: str, chunk_index: int) -> str:
    """Return the deterministic UUIDv5 point id for chunk `chunk_index` of `doc_id`."""
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{doc_id}#{chunk_index}"))


class Chunk(BaseModel):
    """One chunk of a document with the metadata used for filtering and citations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1, max_length=80)
    doc_type: DocType
    chunk_index: int = Field(ge=0, lt=10_000)
    title: str = Field(min_length=1, max_length=400)
    text: str = Field(min_length=1, max_length=20_000)
    platform: list[Platform] = Field(default_factory=list, max_length=len(Platform))
    version: Version | None = None
    version_num: int | None = Field(default=None, ge=0)
    severity: Severity | None = None
    component: Component | None = None
    created_at: str | None = Field(default=None, max_length=40)
    duplicate_of: OptionalBugId = None
    related_bug: OptionalBugId = None

    @property
    def point_id(self) -> str:
        """Deterministic Qdrant point id, so re-ingesting overwrites instead of duplicating."""
        return chunk_point_id(self.doc_id, self.chunk_index)

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder: the title followed by the chunk text."""
        return f"{self.title}\n{self.text}"
