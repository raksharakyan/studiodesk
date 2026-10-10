"""Metric floors for `--strict` runs (`evals/thresholds.json`).

The file maps a dotted metric path to its minimum: the first segment is the suite, the
rest walks the suite's metrics dict (e.g. `duplicates.heldout.f1`). Only suites that ran
are checked, and metrics are computed over completed (non-errored) cases. A metric that
is missing or None (no completed cases) counts as below its floor. Checks are always
reported; they only change the exit code with `--strict`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.suites import SuiteResult

THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.json"


@dataclass(frozen=True)
class ThresholdCheck:
    """One floor and the value it was compared against."""

    name: str
    floor: float
    value: float | None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.value >= self.floor


def load_thresholds(path: Path) -> dict[str, float]:
    """The `floors` mapping of `path`, or {} if the file does not exist.

    Raises:
        ValueError: if the file is not valid JSON or a floor is not a number in [0, 1].
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: invalid JSON") from exc
    floors = data.get("floors") if isinstance(data, dict) else None
    if not isinstance(floors, dict):
        raise ValueError(f"{path.name}: expected an object with a 'floors' mapping")
    result: dict[str, float] = {}
    for name, floor in floors.items():
        if isinstance(floor, bool) or not isinstance(floor, int | float) or not 0 <= floor <= 1:
            raise ValueError(f"{path.name}: floor for {name!r} must be a number in [0, 1]")
        if "." not in name:
            raise ValueError(f"{path.name}: {name!r} must be '<suite>.<metric path>'")
        result[name] = float(floor)
    return result


def metric_value(metrics: dict[str, Any], path: str) -> float | None:
    """Walk `path` (dot-separated) through nested metric dicts; None if absent."""
    node: Any = metrics
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) and not isinstance(node, bool) else None


def check_thresholds(
    results: dict[str, SuiteResult], floors: dict[str, float]
) -> list[ThresholdCheck]:
    """One check per floor whose suite ran, in file order."""
    checks: list[ThresholdCheck] = []
    for name, floor in floors.items():
        suite, path = name.split(".", 1)
        if suite in results:
            checks.append(ThresholdCheck(name, floor, metric_value(results[suite].metrics, path)))
    return checks
