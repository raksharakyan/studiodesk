"""Preliminary M2 eval: does retrieval surface the original of each labelled duplicate?

For every bug report with `duplicate_of`, the duplicate's title + description is used as the
query against the fully ingested dataset (real embedding model, in-memory Qdrant). The query
bug's own chunk is excluded and the original counts as a hit if its doc_id appears among the
top-k remaining chunks. Labels come straight from `data/synthetic/bug_reports.json`.

Run: `uv run python evals/duplicate_retrieval.py` (writes evals/reports/m2_duplicate_retrieval.md).
Reads only Settings defaults (`_env_file=None`); never touches Qdrant Cloud.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from studiodesk.config import Settings
from studiodesk.data.loader import Dataset, load_dataset
from studiodesk.embeddings import Embedder, SentenceTransformerEmbedder
from studiodesk.ingest.pipeline import ingest
from studiodesk.models.documents import DocType
from studiodesk.models.search import SearchFilters
from studiodesk.vectorstore import QdrantStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthetic"
REPORT_PATH = REPO_ROOT / "evals" / "reports" / "m2_duplicate_retrieval.md"
TOP_K = 5


@dataclass(frozen=True)
class CaseResult:
    """Outcome for one labelled duplicate."""

    query_id: str
    expected_id: str
    retrieved: tuple[str, ...]

    @property
    def rank(self) -> int | None:
        """1-based rank of the original among the retrieved chunks, or None if missing."""
        try:
            return self.retrieved.index(self.expected_id) + 1
        except ValueError:
            return None

    @property
    def hit(self) -> bool:
        """True if the original was retrieved."""
        return self.rank is not None


def duplicate_cases(dataset: Dataset) -> list[tuple[str, str, str]]:
    """Return `(duplicate_id, original_id, query_text)` for each labelled duplicate."""
    return [
        (bug.id, bug.duplicate_of, f"{bug.title}\n{bug.description}")
        for bug in dataset.bug_reports
        if bug.duplicate_of is not None
    ]


def evaluate(
    dataset: Dataset,
    embedder: Embedder,
    store: QdrantStore,
    *,
    top_k: int = TOP_K,
    filters: SearchFilters | None = None,
) -> list[CaseResult]:
    """Query each duplicate and record the top-k doc ids, excluding the query bug itself."""
    results: list[CaseResult] = []
    for query_id, expected_id, text in duplicate_cases(dataset):
        hits = store.search(embedder.embed_query(text), filters, top_k + 1)
        retrieved = [h.chunk.doc_id for h in hits if h.chunk.doc_id != query_id][:top_k]
        results.append(CaseResult(query_id, expected_id, tuple(retrieved)))
    return results


def hit_rate(results: list[CaseResult]) -> float:
    """Fraction of cases whose original was retrieved."""
    return sum(r.hit for r in results) / len(results) if results else 0.0


def hit_rate_at(results: list[CaseResult], k: int) -> float:
    """Fraction of cases whose original was ranked within the first `k` results."""
    if not results:
        return 0.0
    return sum(r.rank is not None and r.rank <= k for r in results) / len(results)


def mean_reciprocal_rank(results: list[CaseResult]) -> float:
    """MRR over the retrieved top-k (a miss contributes 0)."""
    return sum(1 / r.rank for r in results if r.rank) / len(results) if results else 0.0


def render_report(
    primary: list[CaseResult], bugs_only: list[CaseResult], settings: Settings, top_k: int = TOP_K
) -> str:
    """Render the markdown report (deterministic: no timestamps)."""
    lines = [
        "# M2 duplicate retrieval (preliminary)",
        "",
        "Preliminary retrieval check for duplicate detection; the full eval suite is built in M4.",
        "",
        f"- Model: `{settings.embedding_model}` @ `{settings.embedding_model_revision}`",
        "- Store: in-memory Qdrant, whole synthetic dataset ingested "
        "(bugs, crashes, patch notes, docs)",
        "- Query: duplicate's `title + description`; the query bug itself is excluded",
        f"- Cases: {len(primary)} bug reports with `duplicate_of`",
        "- Command: `uv run python evals/duplicate_retrieval.py` "
        "(also run by the `slow` pytest test)",
        "",
        "## Results",
        "",
        f"| Setting | Hits | Hit-rate@{top_k} | Hit-rate@1 | MRR@{top_k} |",
        "|---|---|---|---|---|",
        *(
            f"| {name} | {sum(r.hit for r in rs)}/{len(rs)} | {hit_rate(rs):.3f} "
            f"| {hit_rate_at(rs, 1):.3f} | {mean_reciprocal_rank(rs):.3f} |"
            for name, rs in (
                ("All doc types (primary)", primary),
                ("`doc_types=[bug_report]`", bugs_only),
            )
        ),
        "",
        "## Per case (all doc types)",
        "",
        f"| Duplicate | Original | Rank | Top-{top_k} retrieved |",
        "|---|---|---|---|",
    ]
    for r in primary:
        rank = str(r.rank) if r.rank else "**miss**"
        lines.append(f"| {r.query_id} | {r.expected_id} | {rank} | {', '.join(r.retrieved)} |")
    misses = [r for r in primary if not r.hit]
    lines += ["", "## Misses", ""]
    lines += [f"- {r.query_id} (original {r.expected_id})" for r in misses] or ["- none"]
    lines.append("")
    return "\n".join(lines)


def run(
    embedder: Embedder, settings: Settings, data_dir: Path = DATA_DIR
) -> tuple[list[CaseResult], list[CaseResult]]:
    """Ingest into a fresh in-memory store and evaluate both settings."""
    dataset = load_dataset(data_dir)
    client = QdrantClient(location=":memory:")
    try:
        store = QdrantStore(client, "eval")
        ingest(dataset, embedder, store, settings.embedding_batch_size)
        primary = evaluate(dataset, embedder, store)
        bugs_only = evaluate(
            dataset, embedder, store, filters=SearchFilters(doc_types=[DocType.BUG_REPORT])
        )
    finally:
        client.close()
    return primary, bugs_only


def main() -> int:
    """Load the pinned model, run the eval and write the markdown report."""
    settings = Settings(_env_file=None)
    embedder = SentenceTransformerEmbedder.from_settings(settings)
    primary, bugs_only = run(embedder, settings)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(primary, bugs_only, settings), encoding="utf-8")
    print(f"hit-rate@{TOP_K}: {hit_rate(primary):.3f} -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
