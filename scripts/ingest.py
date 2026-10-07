"""Ingest the dataset into Qdrant: chunk, embed and upsert, then print a report.

Reads Qdrant and embedding settings from the environment / `.env` (see `.env.example`).
Requires QDRANT_URL; re-running is idempotent, `--recreate` drops the collection first.
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from studiodesk.config import Settings
from studiodesk.data.loader import load_dataset
from studiodesk.embeddings import SentenceTransformerEmbedder
from studiodesk.ingest.pipeline import IngestReport, ingest
from studiodesk.logging import configure_logging
from studiodesk.vectorstore import QdrantStore, VectorStoreError, build_qdrant_client

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="dataset directory")
    parser.add_argument(
        "--recreate", action="store_true", help="drop and recreate the collection first"
    )
    return parser.parse_args(argv)


def print_report(report: IngestReport) -> None:
    """Print a human-readable ingestion summary."""
    print(f"Collection: {report.collection}{' (recreated)' if report.recreated else ''}")
    print(f"{'type':<12} {'docs':>5} {'chunks':>7}")
    for doc_type, chunks in report.chunks.items():
        print(f"{doc_type:<12} {report.documents.get(doc_type, 0):>5} {chunks:>7}")
    print(f"{'total':<12} {sum(report.documents.values()):>5} {report.total_chunks:>7}")
    print(f"Points in collection: {report.points_in_collection}")
    print(f"Duration: {report.duration_s:.1f}s")


def main(argv: list[str] | None = None) -> int:
    """Run ingestion against the Qdrant configured in Settings. Returns the exit code."""
    args = parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as exc:
        # Report field names and reasons only: input values could be secrets.
        for error in exc.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in error["loc"])
            print(f"error: invalid setting {field}: {error['msg']}", file=sys.stderr)
        return 2
    if not settings.qdrant_url:
        print(
            "error: QDRANT_URL is not set. Set QDRANT_URL and QDRANT_API_KEY in .env "
            "or the environment (see .env.example).",
            file=sys.stderr,
        )
        return 2
    configure_logging(settings)

    try:
        dataset = load_dataset(args.data_dir)
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        print(f"error: failed to load dataset from {args.data_dir}: {exc}", file=sys.stderr)
        return 1

    embedder = SentenceTransformerEmbedder.from_settings(settings)
    client = build_qdrant_client(settings)
    try:
        report = ingest(
            dataset,
            embedder,
            QdrantStore(
                client,
                settings.qdrant_collection,
                upsert_batch_size=settings.qdrant_upsert_batch_size,
            ),
            settings.embedding_batch_size,
            recreate=args.recreate,
        )
    except VectorStoreError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
