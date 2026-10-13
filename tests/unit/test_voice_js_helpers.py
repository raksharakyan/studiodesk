"""Run the voice.js node test runners in tests/js/ and source-level checks on voice.js.

- voice_helpers.test.mjs: the pure exported helpers (no DOM).
- voice_dom.test.mjs: voice.js loaded with an in-memory DOM, fetch and a fake ElevenLabs SDK
  (tests/js/dom_shim.mjs), driving the real clientTools (pending-proposal guard, arm delay,
  trusted-click confirm, console logging).
- voice_preview.test.mjs: the issue preview (pending banner, scroll and focus, visibility
  guard, server-clock expiry timer) and audio-tag stripping in the transcript.

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
        "min_tests": 36,
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
            "stripAudioTags removes ElevenLabs audio tags",
            "stripAudioTags keeps ids, platforms, versions and non-tag brackets",
            "stripAudioTags known limitation: a short bracketed word inside a word is removed",
            "proposalLifetimeMs is expires_at minus the server Date",
            "proposalLifetimeMs is 0 when expired",
            "proposalLifetimeMs is capped at 2^31-1",
            "proposalLifetimeMs is null for missing or invalid input",
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
    "voice_greeting.test.mjs": {
        "min_tests": 23,
        "required": [
            "buildSessionOverrides(true) is the empty first message",
            "buildSessionOverrides is {} for anything but true",
            "buildSessionOverrides returns a fresh object each call",
            "first start greets (no overrides key); after onConnect the next start is silent",
            "storage that already holds '1' gives a silent first start",
            "a throwing sessionStorage getter still greets once via memory",
            "getItem and setItem that throw still greets once via memory",
            "if startSession rejects, the flag is not set and the next start greets",
            "if onConnect never fires, the flag is not set and the next start greets",
            "the hint shows only on a silent start, after connect",
            "an empty or tag-only transcript line keeps the hint; the first real line clears it",
            "the hint is cleared on disconnect",
            'onDisconnect({reason: "agent"}) never restarts the session',
            'onDisconnect({reason: "error"}) never restarts the session',
        ],
    },
    "voice_preview.test.mjs": {
        "min_tests": 30,
        "required": [
            "banner shows after an agent check with a ready proposal and hides on confirm",
            "banner shows after a form check with a ready proposal and hides on cancel",
            "banner stays hidden for duplicate and disabled results (agent and form)",
            "banner hides when confirm returns 404",
            "banner hides when confirm returns 409",
            "banner hides when confirm returns 410",
            "a check with a ready proposal scrolls to the preview and focuses its heading",
            "a pending second agent call scrolls and focuses the preview again without fetching",
            "guard warns and forces the preview visible when #proposal hidden",
            "without a Date header only the arm timer is scheduled",
            "a Date header schedules the expiry timer from expires_at minus Date",
            "expiry retires the current idle proposal",
            "a stale expiry timer does not retire a newer proposal",
            "expiry does not retire a proposal whose confirm is in flight",
            "agent transcript lines have audio tags stripped; user lines are kept as said",
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


def test_console_messages_are_the_known_fixed_strings(voice_js: str) -> None:
    messages = set(re.findall(r'console\.\w+\(\s*"([^"]*)"', voice_js))
    assert messages == {
        "StudioDesk voice: SDK reported an error",
        "StudioDesk voice: start failed",
        "StudioDesk voice: stop failed",
        "StudioDesk: proposal pending but not visible",
    }
    assert re.search(
        r'console\.warn\("StudioDesk: proposal pending but not visible"\);', voice_js
    ), "the visibility guard warning must be a fixed string with no arguments"


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


def _code_only(js: str) -> str:
    """voice.js with // line comments removed, so prose mentioning storage is not counted."""
    return re.sub(r"(?m)^\s*//.*$", "", js)


def test_confirm_token_never_stored_outside_module_state(voice_js: str) -> None:
    code = _code_only(voice_js)
    sinks = ("localStorage", "indexedDB", "document.cookie", "history.pushState")
    for sink in (*sinks, "history.replaceState"):
        assert sink not in code
    assert not re.search(r"dataset\.\w*token", voice_js, re.I)
    assert not re.search(r"setAttribute\([^)]*token", voice_js, re.I)
    # The token / action id never reach a storage API, even indirectly on the same line.
    for line in code.splitlines():
        if re.search(r"Storage|setItem|getItem|removeItem", line):
            assert not re.search(r"token|action_?id", line, re.I), line


def test_session_storage_holds_only_the_greeted_flag(voice_js: str) -> None:
    """The only allowed web-storage use: the per-tab "greeted" boolean (greet once per tab)."""
    code = _code_only(voice_js)
    assert re.search(r'export const GREETED_STORAGE_KEY = "studiodesk\.voice\.greeted";', code)
    uses = re.findall(r"[\w.]*sessionStorage[^\n;]*", code)
    assert uses == [
        'window.sessionStorage?.getItem(GREETED_STORAGE_KEY) === "1"',
        'window.sessionStorage?.setItem(GREETED_STORAGE_KEY, "1")',
    ]
    # No other storage calls on any object (aliases such as `const s = sessionStorage`).
    assert len(re.findall(r"\.(?:setItem|getItem|removeItem|clear)\(", code)) == 2


# --- greet once per tab: page and agent config must agree ----------------------------


def test_voice_hint_is_a_polite_status_region() -> None:
    html = VOICE_HTML.read_text(encoding="utf-8")
    tags = re.findall(r'<[^>]*\bid="voice-hint"[^>]*>', html)
    assert len(tags) == 1
    [tag] = tags
    assert 'role="status"' in tag
    assert 'aria-live="polite"' in tag


def test_session_override_matches_what_agent_json_allows(voice_js: str) -> None:
    """The page may only send the one override the agent allows (else the session fails)."""
    body = re.search(
        r"export function buildSessionOverrides\(greetedBefore\) \{\n(.*?)\n\}", voice_js, re.S
    )
    assert body is not None
    assert body.group(1).strip() == (
        'return greetedBefore === true ? { agent: { firstMessage: "" } } : {};'
    )
    agent = json.loads((REPO_ROOT / "voice" / "agent.json").read_text(encoding="utf-8"))
    assert agent["client_overrides"] == {"first_message": True}
    # The SDK call gets `overrides` only from buildSessionOverrides, and only when silent.
    assert re.search(r"\.\.\.\(silentStart \? \{ overrides \} : \{\}\)", voice_js)
    assert "const overrides = buildSessionOverrides(hasGreeted());" in voice_js
