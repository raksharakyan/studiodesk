"""scripts/validate_dataset.py run as a real subprocess."""

import json
import subprocess
import sys
from pathlib import Path

from studiodesk.data.loader import BUG_REPORTS_FILE

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_dataset.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and script path
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_cli_exits_zero_on_real_data() -> None:
    result = _run()

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "bug_reports" in result.stdout
    assert result.stderr == ""


def test_cli_accepts_explicit_path(data_dir: Path) -> None:
    result = _run(str(data_dir))

    assert result.returncode == 0, result.stderr


def test_cli_exits_nonzero_on_broken_data(data_copy: Path) -> None:
    bugs = json.loads((data_copy / BUG_REPORTS_FILE).read_text())
    bugs[0]["duplicate_of"] = "BUG-9999"
    (data_copy / BUG_REPORTS_FILE).write_text(json.dumps(bugs))

    result = _run(str(data_copy))

    assert result.returncode == 1
    assert "BUG-9999" in result.stderr
    assert "OK" not in result.stdout


def test_cli_exits_nonzero_on_missing_dir(tmp_path: Path) -> None:
    result = _run(str(tmp_path / "missing"))

    assert result.returncode == 1
    assert "failed to load dataset" in result.stderr
