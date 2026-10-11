"""Run the pure voice.js helpers under node (tests/js/voice_helpers.test.mjs).

Skipped when node is not installed. The runner prints one JSON line per JS test; each
failure is reported with its name and assertion message.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "tests" / "js" / "voice_helpers.test.mjs"
VOICE_JS = REPO_ROOT / "src" / "studiodesk" / "static" / "voice.js"
MIN_JS_TESTS = 20

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def js_results() -> tuple[int, list[dict[str, Any]], str]:
    assert NODE is not None
    proc = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [NODE, str(RUNNER), str(VOICE_JS)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        check=False,
    )
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    return proc.returncode, results, proc.stderr


def test_voice_js_imports_and_helpers_pass(
    js_results: tuple[int, list[dict[str, Any]], str],
) -> None:
    code, results, stderr = js_results
    failures = [f"{r['name']}: {r.get('error')}" for r in results if not r["ok"]]
    assert not failures, "\n".join(failures)
    assert code == 0, stderr
    assert len(results) >= MIN_JS_TESTS, f"only {len(results)} JS tests ran; stderr: {stderr}"


@pytest.mark.parametrize(
    "name",
    [
        "normalisePlatform maps spoken names",
        "extractVersion pulls x.y.z out of speech",
        "buildBugCheckBody validates each field",
        "toAgentAskResult with no sources or bad input",
        "toAgentBugResult never includes confirm_token or action_id",
        "safeIssueUrl rejects other hosts, lookalikes and schemes",
        "friendlyError uses Retry-After for 429 and 503",
    ],
)
def test_required_js_case_ran(js_results: tuple[int, list[dict[str, Any]], str], name: str) -> None:
    _, results, _ = js_results
    assert {"name": name, "ok": True} in results
