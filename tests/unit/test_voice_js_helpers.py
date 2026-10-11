"""Run the voice.js node test runners in tests/js/ and source-level checks on voice.js.

- voice_helpers.test.mjs: the pure exported helpers (no DOM).
- voice_dom.test.mjs: voice.js loaded with an in-memory DOM, fetch and a fake ElevenLabs SDK,
  driving the real clientTools (pending-proposal guard, arm delay, trusted-click confirm,
  console logging).

The node runners are skipped when node is not installed; the source checks always run.
Each runner prints one JSON line per JS test; failures are reported with name and message.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from studiodesk.api.pages import ELEVENLABS_SDK_URL

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_DIR = REPO_ROOT / "tests" / "js"
STATIC_DIR = REPO_ROOT / "src" / "studiodesk" / "static"
VOICE_JS = STATIC_DIR / "voice.js"
VOICE_HTML = STATIC_DIR / "voice.html"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

RUNNERS = {
    "voice_helpers.test.mjs": {
        "min_tests": 28,
        "required": [
            "normalisePlatform maps spoken names",
            "extractVersion pulls x.y.z out of speech",
            "buildBugCheckBody validates each field",
            "toAgentAskResult with no sources or bad input",
            "toAgentBugResult never includes confirm_token or action_id",
            "safeIssueUrl rejects other hosts, lookalikes and schemes",
            "friendlyError uses Retry-After for 429 and 503",
            "ApiError keeps only known codes",
            "detailCode maps the three known details and nothing else",
            "formatWait renders speakable waits",
            "friendlyError session messages for the new 503s",
            "friendlyError never echoes a raw server detail",
            "toAgentPendingResult is the fixed object",
        ],
    },
    "voice_dom.test.mjs": {
        "min_tests": 14,
        "required": [
            "init wires the page and the SDK receives the real clientTools",
            "agent check_bug_report returns the pending result and makes no fetch while pending",
            "an agent check that finishes after a form check made a preview never replaces it",
            "the confirm arm delay is 3000 ms",
            "confirm ignores untrusted clicks and clicks before the arm delay",
            "console output never carries the signed URL or SDK message text",
        ],
    },
}

RunResult = tuple[int, list[dict[str, Any]], str]


def _run(runner: str) -> RunResult:
    assert NODE is not None
    proc = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [NODE, str(JS_DIR / runner), str(VOICE_JS)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        check=False,
    )
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    return proc.returncode, results, proc.stderr


@pytest.fixture(scope="module")
def js_results() -> dict[str, RunResult]:
    if NODE is None:
        pytest.skip("node is not installed")
    return {runner: _run(runner) for runner in RUNNERS}


@needs_node
@pytest.mark.parametrize("runner", list(RUNNERS))
def test_js_runner_passes(js_results: dict[str, RunResult], runner: str) -> None:
    code, results, stderr = js_results[runner]
    failures = [f"{r['name']}: {r.get('error')}" for r in results if not r["ok"]]
    assert not failures, "\n".join(failures)
    assert code == 0, stderr
    minimum = RUNNERS[runner]["min_tests"]
    assert isinstance(minimum, int)
    assert len(results) >= minimum, f"only {len(results)} JS tests ran; stderr: {stderr}"


@needs_node
@pytest.mark.parametrize(
    ("runner", "name"),
    [(runner, name) for runner, spec in RUNNERS.items() for name in spec["required"]],  # type: ignore[union-attr]
)
def test_required_js_case_ran(js_results: dict[str, RunResult], runner: str, name: str) -> None:
    _, results, _ = js_results[runner]
    assert {"name": name, "ok": True} in results


# --- source checks (no node needed) --------------------------------------------------


@pytest.fixture(scope="module")
def voice_js() -> str:
    return VOICE_JS.read_text(encoding="utf-8")


def test_console_calls_log_only_literals_and_err_name(voice_js: str) -> None:
    calls = re.findall(r"console\.\w+\((.*?)\);", voice_js, re.DOTALL)
    assert calls, "expected at least one console call"
    literal = r'"[^"`$\\]*"'
    allowed = re.compile(rf"\s*{literal}(\s*,\s*(errName\(\w+\)|{literal}))*\s*")
    bad = [args for args in calls if not allowed.fullmatch(args)]
    assert not bad, f"console calls with non-literal arguments: {bad}"
    total = len(re.findall(r"\bconsole\.", voice_js))
    assert total == len(calls), "a console call was not parsed (check it by hand)"


def test_err_name_returns_only_a_bounded_name(voice_js: str) -> None:
    match = re.search(r"function errName\(err\) \{(.*?)\n\}", voice_js, re.DOTALL)
    assert match is not None
    body = match.group(1)
    assert "err.name" in body and ".slice(0, 40)" in body
    assert "message" not in body and "stack" not in body


def test_confirm_arm_delay_is_3000(voice_js: str) -> None:
    assert re.findall(r"const CONFIRM_ARM_DELAY_MS = (\d+);", voice_js) == ["3000"]


@pytest.mark.parametrize(
    "sink",
    ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"],
)
def test_no_html_or_code_sinks(voice_js: str, sink: str) -> None:
    assert sink not in voice_js
    assert sink not in VOICE_HTML.read_text(encoding="utf-8")


def test_sdk_script_src_is_exactly_the_pinned_url() -> None:
    srcs = re.findall(r'<script\b[^>]*\bsrc="([^"]+)"', VOICE_HTML.read_text(encoding="utf-8"))
    external = [src for src in srcs if src.startswith("http")]
    assert external == [ELEVENLABS_SDK_URL]


def test_no_inline_scripts_in_html() -> None:
    html = VOICE_HTML.read_text(encoding="utf-8")
    for match in re.finditer(r"<script\b([^>]*)>(.*?)</script>", html, re.DOTALL | re.I):
        assert "src=" in match.group(1)
        assert match.group(2).strip() == ""


def test_confirm_token_never_stored_outside_module_state(voice_js: str) -> None:
    for sink in ("localStorage", "sessionStorage", "document.cookie", "history.pushState"):
        assert sink not in voice_js
    assert not re.search(r"dataset\.\w*token", voice_js, re.I)
    assert not re.search(r"setAttribute\([^)]*token", voice_js, re.I)
