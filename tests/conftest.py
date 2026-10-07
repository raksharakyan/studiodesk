"""Shared pytest fixtures for the StudioDesk test suite."""

import hashlib
import logging
import math
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from studiodesk.config import Settings
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
