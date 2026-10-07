"""scripts/ingest.py: failure paths as a real subprocess, happy path in-process.

Every subprocess runs in an empty temporary directory with a minimal environment, so it
cannot read the repository's real `.env` or any QDRANT_* variables from the shell. The
happy path swaps the script's client factory and embedder loader for an in-memory Qdrant
and the fake embedder (`main()` has no injection parameters), so nothing touches the network.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from qdrant_client import QdrantClient

from studiodesk.config import Settings
from studiodesk.data.loader import Dataset
from studiodesk.embeddings import Embedder
from studiodesk.ingest.chunking import chunk_dataset

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"
FAKE_URL = "https://cluster.invalid"


def _clean_env(**extra: str) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")}  # noqa: S108
    env.update(extra)
    return env


def _run(cwd: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and script path
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        env=_clean_env(**env),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_missing_qdrant_url_exits_2(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.returncode == 2
    assert "QDRANT_URL is not set" in result.stderr
    assert result.stdout == ""


def test_empty_qdrant_url_exits_2(tmp_path: Path) -> None:
    result = _run(tmp_path, QDRANT_URL="   ")

    assert result.returncode == 2
    assert "QDRANT_URL is not set" in result.stderr


def test_invalid_settings_reported_without_values(tmp_path: Path) -> None:
    secret_url = "http://user:hunter2-SECRET@cluster.example.com:6333"  # noqa: S105 - fake
    result = _run(
        tmp_path,
        QDRANT_URL=secret_url,
        QDRANT_API_KEY="qk-SECRET-KEY-should-not-print",
        LOG_LEVEL="sk-ant-SECRET-LOOKING-LEVEL",
    )

    assert result.returncode == 2
    assert "invalid setting qdrant_url" in result.stderr
    assert "invalid setting log_level" in result.stderr
    combined = result.stdout + result.stderr
    for leaked in ("hunter2", "SECRET", secret_url, "cluster.example.com"):
        assert leaked not in combined
    assert "Traceback" not in combined


def test_dotenv_in_cwd_is_what_the_script_reads(tmp_path: Path) -> None:
    """Sanity check of the isolation: the script reads `.env` from its working directory."""
    (tmp_path / ".env").write_text("QDRANT_URL=ftp://nope.example\n")

    result = _run(tmp_path)

    assert result.returncode == 2
    assert "invalid setting qdrant_url" in result.stderr


def test_bad_data_dir_exits_1_before_connecting(tmp_path: Path) -> None:
    result = _run(tmp_path, "--data-dir", str(tmp_path / "missing"), QDRANT_URL=FAKE_URL)

    assert result.returncode == 1
    assert "failed to load dataset" in result.stderr


def test_help(tmp_path: Path) -> None:
    result = _run(tmp_path, "--help")

    assert result.returncode == 0
    assert "--recreate" in result.stdout


# --- in-process happy path ---------------------------------------------------------------


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """Import scripts/ingest.py fresh, isolated from the repo `.env` and shell QDRANT_* vars."""
    for name in list(os.environ):
        if name.startswith(("QDRANT_", "EMBEDDING_", "LOG_", "APP_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", FAKE_URL)
    spec = importlib.util.spec_from_file_location("ingest_script_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_collaborators(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, embedder: Embedder, client: Any
) -> list[Settings]:
    seen: list[Settings] = []

    class _Loader:
        @staticmethod
        def from_settings(settings: Settings) -> Embedder:
            seen.append(settings)
            return embedder

    def _client(settings: Settings) -> Any:
        seen.append(settings)
        return client

    monkeypatch.setattr(module, "SentenceTransformerEmbedder", _Loader)
    monkeypatch.setattr(module, "build_qdrant_client", _client)
    return seen


def test_main_happy_path_prints_report(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: Embedder,
    data_dir: Path,
    dataset: Dataset,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = QdrantClient(location=":memory:")
    seen = _patch_collaborators(monkeypatch, script, fake_embedder, client)

    code = script.main(["--data-dir", str(data_dir), "--recreate"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Collection: studiodesk (recreated)" in out
    for doc_type in ("bug_report", "crash_log", "patch_note", "doc"):
        assert doc_type in out
    assert f"Points in collection: {len(chunk_dataset(dataset))}" in out
    assert all(s.qdrant_url == FAKE_URL for s in seen)
    # main() closes the client it built.
    with pytest.raises(RuntimeError):
        client.get_collections()


def test_main_store_error_exits_1(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: Embedder,
    data_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _BrokenClient:
        closed = False

        def collection_exists(self, name: str) -> bool:
            raise ConnectionError("cluster unreachable")

        def close(self) -> None:
            type(self).closed = True

    _patch_collaborators(monkeypatch, script, fake_embedder, _BrokenClient())

    code = script.main(["--data-dir", str(data_dir)])

    assert code == 1
    err = capsys.readouterr().err
    assert (
        err.strip() == "error: VectorStoreError: qdrant ensure_collection failed: ConnectionError"
    )
    assert _BrokenClient.closed is True


def test_main_store_error_never_prints_cause_details(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: Embedder,
    data_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    leaky = "https://admin:qk-SECRET@cluster.internal.example:6333"

    class _LeakyClient:
        def collection_exists(self, name: str) -> bool:
            raise ConnectionError(f"cannot reach {leaky}")

        def close(self) -> None:
            pass

    _patch_collaborators(monkeypatch, script, fake_embedder, _LeakyClient())

    assert script.main(["--data-dir", str(data_dir)]) == 1

    captured = capsys.readouterr()
    for fragment in ("qk-SECRET", "admin", "cluster.internal"):
        assert fragment not in captured.err + captured.out


class _CountingClient:
    """Wraps an in-memory client and records the size of every upsert request."""

    def __init__(self) -> None:
        self._inner = QdrantClient(location=":memory:")
        self.upserts: list[int] = []

    def upsert(self, collection_name: str, points: Any, **kwargs: Any) -> Any:
        self.upserts.append(len(points))
        return self._inner.upsert(collection_name, points=points, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_main_uses_configured_upsert_batch_size(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: Embedder,
    data_dir: Path,
    dataset: Dataset,
) -> None:
    monkeypatch.setenv("QDRANT_UPSERT_BATCH_SIZE", "50")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "1024")  # one embed batch -> one store.upsert
    client = _CountingClient()
    _patch_collaborators(monkeypatch, script, fake_embedder, client)

    assert script.main(["--data-dir", str(data_dir)]) == 0

    total = len(chunk_dataset(dataset))
    assert client.upserts == [50] * (total // 50) + ([total % 50] if total % 50 else [])
