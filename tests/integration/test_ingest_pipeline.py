"""Ingest pipeline and QdrantStore against an in-memory Qdrant with the fake embedder."""

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from studiodesk.data.loader import Dataset
from studiodesk.embeddings import Embedder
from studiodesk.ingest.chunking import chunk_dataset
from studiodesk.ingest.pipeline import batched, ingest
from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import DocType, version_to_int
from studiodesk.models.search import SearchFilters
from studiodesk.vectorstore import QdrantStore, VectorStoreError


@pytest.fixture
def mem_client() -> Iterator[QdrantClient]:
    client = QdrantClient(location=":memory:")
    yield client
    client.close()


def _ticker(*values: float) -> object:
    it = iter(values)
    return lambda: next(it)


# --- pipeline ----------------------------------------------------------------------------


def test_ingest_whole_dataset_counts_match(
    dataset: Dataset, fake_embedder: Embedder, store: QdrantStore
) -> None:
    report = ingest(dataset, fake_embedder, store, batch_size=16, clock=_ticker(10.0, 12.5))

    expected_chunks = len(chunk_dataset(dataset))
    assert report.documents == {
        "bug_report": len(dataset.bug_reports),
        "crash_log": len(dataset.crash_logs),
        "patch_note": len(dataset.patch_notes),
        "doc": len(dataset.docs),
    }
    assert report.chunks["bug_report"] == len(dataset.bug_reports)
    assert report.chunks["crash_log"] == len(dataset.crash_logs)
    assert report.chunks["patch_note"] == sum(1 + len(p.fixes) for p in dataset.patch_notes)
    assert sum(report.chunks.values()) == report.total_chunks == expected_chunks
    assert report.points_in_collection == expected_chunks == store.count()
    assert report.duration_s == 2.5
    assert report.collection == "test"
    assert report.recreated is False


@pytest.mark.parametrize("batch_size", [1, 7, 1000])
def test_ingest_is_idempotent_for_any_batch_size(
    dataset: Dataset, fake_embedder: Embedder, store: QdrantStore, batch_size: int
) -> None:
    first = ingest(dataset, fake_embedder, store, batch_size=batch_size)
    second = ingest(dataset, fake_embedder, store, batch_size=batch_size)

    assert second.points_in_collection == first.points_in_collection == first.total_chunks


def test_recreate_drops_stale_points(
    dataset: Dataset, fake_embedder: Embedder, store: QdrantStore
) -> None:
    stale = Chunk(doc_id="STALE", doc_type=DocType.DOC, chunk_index=0, title="t", text="old")
    store.ensure_collection(fake_embedder.dim)
    store.upsert([stale], fake_embedder.embed_documents(["old"]))

    report = ingest(dataset, fake_embedder, store, batch_size=64, recreate=True)

    assert report.recreated is True
    assert report.points_in_collection == report.total_chunks


def test_reingest_without_recreate_keeps_stale_points(
    dataset: Dataset, fake_embedder: Embedder, store: QdrantStore
) -> None:
    stale = Chunk(doc_id="STALE", doc_type=DocType.DOC, chunk_index=0, title="t", text="old")
    store.ensure_collection(fake_embedder.dim)
    store.upsert([stale], fake_embedder.embed_documents(["old"]))

    report = ingest(dataset, fake_embedder, store, batch_size=64)

    assert report.points_in_collection == report.total_chunks + 1


def test_empty_dataset_ingests_nothing(fake_embedder: Embedder, store: QdrantStore) -> None:
    report = ingest(Dataset(bug_reports=[], crash_logs=[], patch_notes=[]), fake_embedder, store, 8)

    assert report.total_chunks == report.points_in_collection == 0
    assert set(report.chunks.values()) == {0}


@pytest.mark.parametrize("size", [0, -5])
def test_batched_rejects_non_positive_size(size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        list(batched([], size))


def test_batched_slices() -> None:
    assert [list(b) for b in batched([1, 2, 3, 4, 5], 2)] == [[1, 2], [3, 4], [5]]  # type: ignore[list-item]


def test_payload_stores_every_metadata_field(ingested_store: QdrantStore, dataset: Dataset) -> None:
    hits = ingested_store.search([1.0] * 64, SearchFilters(doc_types=["crash_log"]), top_k=100)
    crash = next(c for c in dataset.crash_logs if c.related_bug)

    [hit] = [h for h in hits if h.chunk.doc_id == crash.id]
    assert hit.chunk.related_bug == crash.related_bug
    assert hit.chunk.version_num == version_to_int(crash.version)
    assert hit.chunk.platform == [crash.platform]
    assert hit.chunk.severity == crash.severity
    assert hit.chunk.component == crash.component


# --- filters restrict results ------------------------------------------------------------


def _all(store: QdrantStore, filters: SearchFilters) -> list[Chunk]:
    vec = [1.0] + [0.0] * 63
    return [h.chunk for h in store.search(vec, filters, top_k=1000)]


def _expected(dataset: Dataset, predicate: object) -> set[tuple[str, int]]:
    return {(c.doc_id, c.chunk_index) for c in chunk_dataset(dataset) if predicate(c)}  # type: ignore[operator]


@pytest.mark.parametrize(
    ("filters", "predicate"),
    [
        (SearchFilters(platforms=["ps5"]), lambda c: "ps5" in c.platform),
        (SearchFilters(platforms=["switch", "pc"]), lambda c: {"switch", "pc"} & set(c.platform)),
        (SearchFilters(severities=["critical"]), lambda c: c.severity == "critical"),
        (SearchFilters(components=["save_system"]), lambda c: c.component == "save_system"),
        (SearchFilters(doc_types=["doc"]), lambda c: c.doc_type == "doc"),
        (
            SearchFilters(version_min="1.1.0", version_max="1.2.0"),
            lambda c: c.version_num is not None and 10100 <= c.version_num <= 10200,
        ),
        (
            SearchFilters(version_min="1.3.0"),
            lambda c: c.version_num is not None and c.version_num >= 10300,
        ),
        (
            SearchFilters(
                doc_types=["bug_report"], platforms=["pc"], severities=["high", "critical"]
            ),
            lambda c: (
                c.doc_type == "bug_report"
                and "pc" in c.platform
                and c.severity in ("high", "critical")
            ),
        ),
    ],
)
def test_filters_return_exactly_matching_chunks(
    ingested_store: QdrantStore, dataset: Dataset, filters: SearchFilters, predicate: object
) -> None:
    got = {(c.doc_id, c.chunk_index) for c in _all(ingested_store, filters)}
    expected = _expected(dataset, predicate)

    assert expected, "filter case should match something in the dataset"
    assert got == expected


def test_version_filter_excludes_unversioned_docs(ingested_store: QdrantStore) -> None:
    chunks = _all(ingested_store, SearchFilters(version_max="99.99.99"))

    assert chunks
    assert all(c.doc_type is not DocType.DOC for c in chunks)


def test_no_match_returns_empty(ingested_store: QdrantStore) -> None:
    filters = SearchFilters(doc_types=["doc"], severities=["critical"])

    assert _all(ingested_store, filters) == []


def test_search_orders_by_score_and_respects_top_k(
    ingested_store: QdrantStore, fake_embedder: Embedder
) -> None:
    hits = ingested_store.search(fake_embedder.embed_query("save corrupted cryo"), None, top_k=7)

    assert len(hits) == 7
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


# --- store edge cases --------------------------------------------------------------------


def test_ensure_collection_dimension_mismatch(store: QdrantStore) -> None:
    store.ensure_collection(64)
    store.ensure_collection(64)  # idempotent

    with pytest.raises(VectorStoreError, match="vector size 64, expected 32"):
        store.ensure_collection(32)

    store.ensure_collection(32, recreate=True)
    assert store.count() == 0


def test_upsert_length_mismatch(store: QdrantStore) -> None:
    store.ensure_collection(64)
    chunk = Chunk(doc_id="X", doc_type=DocType.DOC, chunk_index=0, title="t", text="x")

    with pytest.raises(ValueError, match="same length"):
        store.upsert([chunk], [])
    store.upsert([], [])
    assert store.count() == 0


def test_search_on_missing_collection_raises_store_error(store: QdrantStore) -> None:
    with pytest.raises(VectorStoreError) as excinfo:
        store.search([0.0] * 64, None, 5)
    with pytest.raises(VectorStoreError):
        store.count()

    assert str(excinfo.value).startswith("qdrant search failed")


def test_search_skips_points_with_invalid_payload(
    mem_client: QdrantClient, fake_embedder: Embedder, caplog: pytest.LogCaptureFixture
) -> None:
    store = QdrantStore(mem_client, "bad")
    store.ensure_collection(fake_embedder.dim)
    good = Chunk(doc_id="GOOD", doc_type=DocType.DOC, chunk_index=0, title="t", text="x")
    store.upsert([good], fake_embedder.embed_documents(["x"]))
    mem_client.upsert(
        "bad",
        points=[
            qm.PointStruct(
                id=1, vector=fake_embedder.embed_query("x"), payload={"doc_id": "B", "evil": 1}
            )
        ],
    )

    hits = store.search(fake_embedder.embed_query("x"), None, top_k=10)

    assert [h.chunk.doc_id for h in hits] == ["GOOD"]
    assert "skipping point with invalid payload" in caplog.text
