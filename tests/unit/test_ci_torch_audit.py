"""The CI step that audits the public torch release extracts the right version.

pip-audit skips local versions like `2.14.1+cpu`, so CI strips the local part with a sed
expression. This runs that exact expression (read from ci.yml) against sample lines.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

CI_FILE = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(shutil.which("sed") is None, reason="sed not available")


def _sed_expression() -> str:
    match = re.search(r"sed -nE '([^']+)'", CI_FILE.read_text(encoding="utf-8"))
    assert match, "torch audit sed expression not found in ci.yml"
    return match.group(1)


def _extract(text: str) -> list[str]:
    sed = shutil.which("sed")
    assert sed is not None
    result = subprocess.run(  # noqa: S603 - fixed binary, expression from our own CI file
        [sed, "-nE", _sed_expression()],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return sorted(set(result.stdout.split()))


@pytest.mark.parametrize(
    "line",
    [
        "torch==2.14.1+cpu ; sys_platform == 'linux'",
        "torch==2.14.1 ; sys_platform != 'linux'",
        "torch==2.14.1+cpu",
        "torch==2.14.1",
        "torch==2.14.1 \\",
    ],
)
def test_extracts_public_version(line: str) -> None:
    assert _extract(line + "\n") == ["2.14.1"]


def test_both_platform_lines_dedupe_to_one_version() -> None:
    export = (
        "fastapi==0.142.2\n"
        "torch==2.14.1+cpu ; sys_platform == 'linux'\n"
        "torch==2.14.1 ; sys_platform != 'linux'\n"
        "torchvision==0.30.0\n"
        "    # via torch\n"
        "pytorch-lightning==3.0.0\n"
    )

    assert _extract(export) == ["2.14.1"]


def test_no_torch_extracts_nothing() -> None:
    assert _extract("fastapi==0.142.2\ntorchaudio==2.14.1\n") == []
