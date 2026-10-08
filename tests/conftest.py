"""Shared pytest fixtures for the StudioDesk test suite."""

import hashlib
import logging
import math
import os
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from studiodesk.config import Settings
from studiodesk.data.loader import Dataset, load_dataset
from studiodesk.ingest.pipeline import ingest
from studiodesk.main import create_app
from studiodesk.vectorstore import QdrantStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthetic"


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no model download, stable across runs."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in text.lower().split():
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest) % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers (pyproject.toml is owned by dev-agent)."""
    config.addinivalue_line(
        "markers", "slow: loads the real embedding model; deselect with -m 'not slow'"
    )
    config.addinivalue_line(
        "markers",
        "live: calls real external services; skipped unless STUDIODESK_LIVE=1 is set",
    )


LIVE_ENV_VAR = "STUDIODESK_LIVE"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `@pytest.mark.live` tests unless STUDIODESK_LIVE=1 (never in the default run)."""
    if os.environ.get(LIVE_ENV_VAR) == "1":
        return
    skip_live = pytest.mark.skip(reason=f"live test: set {LIVE_ENV_VAR}=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Undo handler/level changes made by `configure_logging` during a test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def settings() -> Settings:
    """Deterministic test Settings, isolated from any real `.env` file."""
    return Settings(_env_file=None, app_env="test", app_version="9.9.9")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Deterministic test embedder."""
    return FakeEmbedder()


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    """Empty vector store backed by an in-memory Qdrant client."""
    client = QdrantClient(location=":memory:")
    yield QdrantStore(client, "test")
    client.close()


@pytest.fixture
def app(settings: Settings, fake_embedder: FakeEmbedder, store: QdrantStore) -> FastAPI:
    """App built from the injected test settings, fake embedder and in-memory store."""
    return create_app(settings, embedder=fake_embedder, store=store)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """TestClient for the test app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def data_dir() -> Path:
    """Path to the real synthetic dataset."""
    return DATA_DIR


@pytest.fixture
def data_copy(tmp_path: Path) -> Path:
    """A writable copy of the real dataset for building broken variants."""
    target = tmp_path / "synthetic"
    shutil.copytree(DATA_DIR, target)
    return target


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """The real synthetic dataset, parsed once per session (models are frozen)."""
    return load_dataset(DATA_DIR)


@pytest.fixture
def ingested_store(
    store: QdrantStore, fake_embedder: FakeEmbedder, dataset: Dataset
) -> QdrantStore:
    """In-memory store with the whole dataset ingested via the fake embedder."""
    ingest(dataset, fake_embedder, store, batch_size=32)
    return store
