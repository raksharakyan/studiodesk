"""Shared pytest fixtures for the StudioDesk test suite."""

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studiodesk.config import Settings
from studiodesk.main import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthetic"


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
def app(settings: Settings) -> FastAPI:
    """App built from the injected test settings."""
    return create_app(settings)


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
