"""Ingestion pipeline: chunk the dataset, embed in batches, upsert into the vector store.

Idempotent: point ids are derived from `doc_id#chunk_index`, so running it again on the
same dataset overwrites the same points instead of adding new ones.
"""

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence

from pydantic import BaseModel, ConfigDict

from studiodesk.data.loader import Dataset
from studiodesk.embeddings import Embedder
from studiodesk.ingest.chunking import chunk_dataset
from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import DocType
from studiodesk.vectorstore import QdrantStore

logger = logging.getLogger(__name__)


class IngestReport(BaseModel):
    """Outcome of one ingestion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: str
    recreated: bool
    documents: dict[str, int]
    chunks: dict[str, int]
    total_chunks: int
    points_in_collection: int
    duration_s: float


def batched(chunks: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    """Yield consecutive slices of at most `size` chunks."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def document_counts(dataset: Dataset) -> dict[str, int]:
    """Number of source documents per doc type."""
    return {
        DocType.BUG_REPORT.value: len(dataset.bug_reports),
        DocType.CRASH_LOG.value: len(dataset.crash_logs),
        DocType.PATCH_NOTE.value: len(dataset.patch_notes),
        DocType.DOC.value: len(dataset.docs),
    }


def ingest(
    dataset: Dataset,
    embedder: Embedder,
    store: QdrantStore,
    batch_size: int,
    *,
    recreate: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> IngestReport:
    """Chunk, embed and upsert the whole dataset, returning counts and timing.

    Args:
        dataset: Parsed dataset to ingest.
        embedder: Produces document vectors; its `dim` sizes the collection.
        store: Target vector store.
        batch_size: Chunks embedded and upserted per batch.
        recreate: Drop and recreate the collection first.
        clock: Monotonic clock in seconds (injectable for tests).

    Raises:
        VectorStoreError: if the store fails.
    """
    started = clock()
    chunks = chunk_dataset(dataset)
    store.ensure_collection(embedder.dim, recreate=recreate)
    for batch in batched(chunks, batch_size):
        vectors = embedder.embed_documents([chunk.embedding_text for chunk in batch])
        store.upsert(batch, vectors)
    per_type = Counter(chunk.doc_type.value for chunk in chunks)
    report = IngestReport(
        collection=store.collection,
        recreated=recreate,
        documents=document_counts(dataset),
        chunks={doc_type.value: per_type.get(doc_type.value, 0) for doc_type in DocType},
        total_chunks=len(chunks),
        points_in_collection=store.count(),
        duration_s=round(clock() - started, 3),
    )
    logger.info(
        "ingestion finished",
        extra={"total_chunks": report.total_chunks, "duration_s": report.duration_s},
    )
    return report
