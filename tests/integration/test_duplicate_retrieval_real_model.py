"""Real-model retrieval check (slow): labelled duplicates retrieve their originals.

Loads the pinned MiniLM model from the Hugging Face cache (offline when the pinned
snapshot is cached; CI warms the cache), ingests everything into in-memory Qdrant and
writes evals/reports/m2_duplicate_retrieval.md. The true hit-rate is always written to the
report; the assertion is only a regression floor.
"""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

from studiodesk.config import Settings
from studiodesk.data.loader import Dataset
from studiodesk.embeddings import SentenceTransformerEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_MODULE = REPO_ROOT / "evals" / "duplicate_retrieval.py"
HIT_RATE_FLOOR = 0.8  # measured 16/16 = 1.0 on 2026-10-07; floor allows minor CPU drift
EXPECTED_CASES = 16

pytestmark = pytest.mark.slow


def _load_eval() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m2_duplicate_retrieval", EVAL_MODULE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned_snapshot_cached(settings: Settings) -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = "models--" + settings.embedding_model.replace("/", "--")
    return (hf_home / "hub" / repo_dir / "snapshots" / settings.embedding_model_revision).is_dir()


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    settings = Settings(_env_file=None)
    mp = pytest.MonkeyPatch()
    if _pinned_snapshot_cached(settings):
        mp.setenv("HF_HUB_OFFLINE", "1")  # guarantee no network when the model is cached
    try:
        return SentenceTransformerEmbedder.from_settings(settings)
    finally:
        mp.undo()


def test_real_embedder_shape(real_embedder: SentenceTransformerEmbedder) -> None:
    a, b, c = real_embedder.embed_documents(
        ["save lost after cryo sleep", "cryo pod wiped my save", "rover horn too loud"]
    )

    assert real_embedder.dim == 384
    assert len(a) == 384
    assert abs(sum(x * x for x in a) - 1.0) < 1e-4

    def cos(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=True))

    assert cos(a, b) > cos(a, c)


def test_duplicates_retrieve_original_in_top5(
    real_embedder: SentenceTransformerEmbedder, dataset: Dataset
) -> None:
    ev = _load_eval()
    settings = Settings(_env_file=None)

    primary, bugs_only = ev.run(real_embedder, settings)
    ev.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ev.REPORT_PATH.write_text(ev.render_report(primary, bugs_only, settings), encoding="utf-8")

    assert len(primary) == EXPECTED_CASES == len(ev.duplicate_cases(dataset))
    assert all(r.query_id not in r.retrieved for r in primary), "query doc is excluded"
    assert all(len(r.retrieved) == ev.TOP_K for r in primary)
    misses = [(r.query_id, r.expected_id) for r in primary if not r.hit]
    rate = ev.hit_rate(primary)
    assert rate >= HIT_RATE_FLOOR, f"hit-rate@5 {rate:.3f} below floor; misses: {misses}"
