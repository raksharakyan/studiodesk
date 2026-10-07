"""Minimal POST /search smoke test (fake embedder + in-memory Qdrant)."""

from pathlib import Path

from fastapi.testclient import TestClient

from studiodesk.data.loader import load_dataset
from studiodesk.embeddings import Embedder
from studiodesk.ingest.pipeline import ingest
from studiodesk.vectorstore import QdrantStore


def test_search_returns_filtered_hits(
    client: TestClient, fake_embedder: Embedder, store: QdrantStore, data_dir: Path
) -> None:
    ingest(load_dataset(data_dir), fake_embedder, store, batch_size=32)

    response = client.post(
        "/search",
        json={"query": "save file corrupted after cryo sleep", "filters": {"platforms": ["ps5"]}},
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert 0 < len(hits) <= 5
    assert all("ps5" in hit["metadata"]["platform"] for hit in hits)


def test_search_without_collection_returns_generic_503(client: TestClient) -> None:
    response = client.post("/search", json={"query": "anything"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Search is temporarily unavailable"}
