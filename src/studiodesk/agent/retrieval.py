"""`Retriever`: embed a text and search the vector store, optionally excluding documents."""

from collections.abc import Collection

from studiodesk.embeddings import Embedder
from studiodesk.models.documents import DocType
from studiodesk.models.search import SearchFilters
from studiodesk.vectorstore import QdrantStore, ScoredChunk

BUG_REPORTS_ONLY = SearchFilters(doc_types=[DocType.BUG_REPORT])


class Retriever:
    """Pairs an embedder with a vector store. Both are injected and owned by the caller."""

    def __init__(self, embedder: Embedder, store: QdrantStore) -> None:
        """Use `embedder` for queries and `store` for similarity search."""
        self._embedder = embedder
        self._store = store

    def search(
        self,
        text: str,
        filters: SearchFilters | None,
        top_k: int,
        *,
        exclude_ids: Collection[str] = (),
    ) -> list[ScoredChunk]:
        """Return up to `top_k` chunks most similar to `text`, skipping `exclude_ids` docs.

        Over-fetches by `len(exclude_ids)` so exclusions do not shrink the result for
        single-chunk documents such as bug reports.

        Raises:
            VectorStoreError: if the store fails.
        """
        vector = self._embedder.embed_query(text)
        hits = self._store.search(vector, filters, top_k + len(exclude_ids))
        return [hit for hit in hits if hit.chunk.doc_id not in exclude_ids][:top_k]

    def search_bug_reports(
        self, text: str, top_k: int, *, exclude_ids: Collection[str] = ()
    ) -> list[ScoredChunk]:
        """Like `search`, restricted to `doc_type == bug_report`."""
        return self.search(text, BUG_REPORTS_ONLY, top_k, exclude_ids=exclude_ids)
