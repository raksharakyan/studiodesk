"""Validate the synthetic dataset: print counts and exit non-zero on any error."""

import argparse
import sys
from pathlib import Path

from studiodesk.data.loader import validate_dataset

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def main(argv: list[str] | None = None) -> int:
    """Run validation and print a human-readable report. Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    report = validate_dataset(args.data_dir)
    print(f"Dataset: {args.data_dir}")
    for kind, count in report.counts.items():
        print(f"  {kind:<12} {count}")
    if report.severity_counts:
        print("Severity (bug reports + crash logs):")
        for severity, count in report.severity_counts.items():
            print(f"  {severity:<12} {count}")
    if report.errors:
        print(f"{len(report.errors)} error(s):", file=sys.stderr)
        for error in report.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
