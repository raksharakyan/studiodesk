"""Static checks on the recorded GitHub Pages showcase in site/ (no backend).

- site/index.html: the meta CSP and referrer policy, no inline scripts, styles or handlers,
  only relative same-origin script/style paths, accessibility basics, and every element id
  showcase.js looks up exists.
- site/showcase.js: no HTML or code sinks, no storage or console, and the only network call
  is the relative, same-origin fetch of data/showcase.json.
- site/data/showcase.json: matches the real /ask and /bugs/check response models, with no
  confirm tokens, and a filed-issue link into the sandbox repo.
- tests/js/showcase.test.mjs (node): the pure helpers and a hostile-data render.
"""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from studiodesk.models.answer import AnswerResponse
from studiodesk.models.bugs import BugCheckRequest, BugCheckResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = REPO_ROOT / "site"
INDEX_HTML = SITE_DIR / "index.html"
SHOWCASE_JS = SITE_DIR / "showcase.js"
SITE_CSS = SITE_DIR / "assets" / "site.css"
SHOWCASE_JSON = SITE_DIR / "data" / "showcase.json"
JS_RUNNER = REPO_ROOT / "tests" / "js" / "showcase.test.mjs"
NODE = shutil.which("node")

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'"
)
SANDBOX_ISSUES_PREFIX = "https://github.com/raksharakyan/starfall-outpost-issues/issues/"
ALLOWED_LINK_PREFIXES = (
    "https://github.com/raksharakyan/studiodesk",
    "https://github.com/raksharakyan/starfall-outpost-issues",
)


class Page(HTMLParser):
    """Collects tags, attributes, inline script text and style blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.inline_scripts: list[str] = []
        self.style_blocks = 0
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"
        if tag == "style":
            self.style_blocks += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline_scripts.append(data)

    def find(self, tag: str) -> list[dict[str, str | None]]:
        return [attrs for name, attrs in self.tags if name == tag]

    def ids(self) -> list[str]:
        return [attrs["id"] or "" for _, attrs in self.tags if "id" in attrs]


@pytest.fixture(scope="module")
def page() -> Page:
    parsed = Page()
    parsed.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return parsed


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return SHOWCASE_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def code(js: str) -> str:
    """showcase.js without comments, so prose is not matched."""
    stripped = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", stripped)


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    loaded = json.loads(SHOWCASE_JSON.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


# --- index.html --------------------------------------------------------------------------


def test_meta_csp_and_referrer(page: Page) -> None:
    metas = page.find("meta")
    csp = [m for m in metas if (m.get("http-equiv") or "").lower() == "content-security-policy"]
    assert [m.get("content") for m in csp] == [CSP]
    assert "frame-ancestors" not in CSP  # ignored in a meta CSP
    assert {"name": "referrer", "content": "no-referrer"} in metas
    # The CSP meta must come before any script or stylesheet so it applies to them.
    order = [t for t, a in page.tags if t in ("script", "link") or (t == "meta" and a in csp)]
    assert order[0] == "meta"


def test_no_inline_script_style_or_handlers(page: Page, html: str) -> None:
    assert page.inline_scripts == []
    assert page.style_blocks == 0
    for tag, attrs in page.tags:
        assert "style" not in attrs, f"style attribute on <{tag}>"
        handlers = [name for name in attrs if name.startswith("on")]
        assert not handlers, f"inline handler {handlers} on <{tag}>"
    assert "javascript:" not in html.lower()


def test_scripts_and_styles_are_relative_same_origin(page: Page) -> None:
    scripts = page.find("script")
    assert [s.get("src") for s in scripts] == ["showcase.js"]
    assert scripts[0].get("type") == "module"
    sheets = [link.get("href") for link in page.find("link") if link.get("rel") == "stylesheet"]
    assert sheets == ["assets/styles.css", "assets/site.css"]
    for tag, attrs in page.tags:
        for name in ("src", "href", "action", "srcset", "poster", "data"):
            value = attrs.get(name)
            if value is None or tag == "a":
                continue
            assert not value.startswith(("/", "http:", "https:")), f"<{tag} {name}={value!r}>"
            assert value == "data:," or ":" not in value, f"<{tag} {name}={value!r}>"


def test_no_third_party_or_voice_sdk(html: str) -> None:
    lowered = html.lower()
    for needle in (
        "cdn.",
        "jsdelivr",
        "unpkg",
        "elevenlabs/client",
        "integrity=",
        "<iframe",
        "<form",
    ):
        assert needle not in lowered, needle


def test_links_are_known_https_github_with_noopener(page: Page) -> None:
    for attrs in page.find("a"):
        href = attrs.get("href") or ""
        if href.startswith("#"):
            continue
        assert href.startswith(ALLOWED_LINK_PREFIXES), href
        assert set((attrs.get("rel") or "").split()) >= {"noopener", "noreferrer"}, href


def test_banner_copy(html: str) -> None:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))
    assert (
        "This is a recorded demo: every answer below was produced by the real StudioDesk pipeline"
        in text
    )
    assert "Qdrant, MiniLM) on" in text
    assert "To run it live with your own keys, see the GitHub repo." in text
    assert "Live questions need the backend; see Run it yourself in the README" in text
    assert "Recorded voice session (text transcript)" in text


def test_accessibility_basics(page: Page, html: str) -> None:
    assert re.search(r'<html lang="en">', html)
    assert len(page.find("h1")) == 1
    assert len(page.find("main")) == 1
    assert len(page.find("header")) >= 1 and len(page.find("footer")) == 1
    skip = page.find("a")[0]
    assert skip.get("href") == "#main" and skip.get("class") == "skip-link"
    ids = page.ids()
    assert len(ids) == len(set(ids)), "duplicate ids"
    # Status and error regions are announced.
    by_id = {a["id"]: a for _, a in page.tags if a.get("id")}
    for region in ("load-status", "ask-status", "bug-status"):
        assert by_id[region].get("role") == "status" and by_id[region].get("aria-live") == "polite"
    assert by_id["load-error"].get("role") == "alert"
    # Confirm and Cancel exist but are disabled and described by the replay caption.
    for button in ("confirm-btn", "cancel-btn"):
        assert "disabled" in by_id[button]
        assert by_id[button].get("aria-describedby") == "replay-filing"
    # Every button has a type and visible text.
    for attrs in page.find("button"):
        assert attrs.get("type") == "button"


def test_every_id_showcase_js_uses_exists(page: Page, code: str) -> None:
    used = set(re.findall(r'(?:\$|setText|show)\(\s*"([a-z0-9-]+)"', code))
    used |= set(re.findall(r'getElementById\(\s*"([a-z0-9-]+)"', code))
    used |= set(re.findall(r'"(count-[a-z]+)"', code))
    assert used, "no ids found in showcase.js"
    missing = used - set(page.ids())
    assert not missing, f"ids used by showcase.js but missing in index.html: {sorted(missing)}"
    eval_slots = {a.get("data-eval") for _, a in page.tags if "data-eval" in a}
    assert eval_slots == {
        "retrieval_hit_at_5",
        "citation_validity",
        "duplicates_heldout",
        "injection_pass",
        "routing_component_accuracy",
    }
    bug_chips = [a.get("data-bug") for _, a in page.tags if "data-bug" in a]
    assert bug_chips == ["duplicate", "new", "injection"]


# --- showcase.js -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sink",
    [
        r"\.innerHTML\b",
        r"\.outerHTML\b",
        r"insertAdjacentHTML",
        r"document\.write",
        r"\beval\(",
        r"new Function\b",
        r"createContextualFragment",
        r"DOMParser",
        r"\.srcdoc\b",
        r"setTimeout\(\s*[\"'`]",
        r"setInterval\(",
        r"localStorage",
        r"sessionStorage",
        r"indexedDB",
        r"document\.cookie",
        r"\bconsole\.",
        r"XMLHttpRequest",
        r"WebSocket",
        r"EventSource",
        r"sendBeacon",
        r"\bimport\(",
        r"\.style\b",
        r"setAttribute\(\s*[\"'](?:style|on\w+|href|src)",
        r"\bon[a-z]+\s*=",
        r"javascript:",
        r"window\.open",
        r"location\s*=",
        r"postMessage",
    ],
)
def test_showcase_js_has_no_sink(code: str, sink: str) -> None:
    assert not re.search(sink, code), sink


def test_only_fetch_is_the_relative_showcase_url(code: str) -> None:
    calls = re.findall(r"\bfetch\(\s*([^,)]+)", code)
    assert calls == ["SHOWCASE_URL"]
    assert re.search(r'export const SHOWCASE_URL = "data/showcase\.json";', code)
    for opt in ('credentials: "same-origin"', 'redirect: "error"', 'method: "GET"'):
        assert opt in code, opt


def test_only_url_literal_is_the_sandbox_prefix(code: str) -> None:
    urls = re.findall(r"[\"'`](https?://[^\"'`]*)", code)
    assert urls == [SANDBOX_ISSUES_PREFIX]


def test_links_are_only_set_from_the_validated_filed_issue(code: str) -> None:
    assignments = re.findall(r"\.href\s*=\s*([^;]+);", code)
    assert assignments == ["link.href"]
    assert "const link = filedIssueLink(example);" in code


def test_site_css_loads_nothing() -> None:
    css = SITE_CSS.read_text(encoding="utf-8")
    assert "url(" not in css and "@import" not in css


def test_site_has_only_expected_file_types() -> None:
    files = [p for p in SITE_DIR.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    assert {p.suffix for p in files} <= {".html", ".js", ".css", ".json"}
    # styles.css is copied in by scripts/build_site.py, never committed under site/.
    assert not (SITE_DIR / "assets" / "styles.css").exists()


# --- showcase.json -------------------------------------------------------------------------


def test_showcase_top_level_contract(data: dict[str, Any]) -> None:
    assert data["schema"] == 1
    assert set(data) >= {
        "recorded_at",
        "models",
        "dataset",
        "questions",
        "bug_checks",
        "file_anyway",
        "filed_issue_example",
        "voice_transcript",
        "evals",
    }
    assert set(data["models"]) >= {"llm", "embedding"}
    assert set(data["dataset"]) >= {"bug_reports", "crash_logs", "patch_notes", "docs"}
    assert set(data["evals"]) >= {
        "retrieval_hit_at_5",
        "citation_validity",
        "duplicates_heldout_precision",
        "duplicates_heldout_recall",
        "injection_pass",
        "routing_component_accuracy",
    }
    assert re.fullmatch(r"[0-9]{1,4}/[0-9]{1,4}", data["evals"]["injection_pass"])
    ratios = [v for k, v in data["evals"].items() if k != "injection_pass"]
    assert all(isinstance(v, int | float) and 0 <= v <= 1 for v in ratios)


def test_questions_are_real_ask_responses(data: dict[str, Any]) -> None:
    assert len(data["questions"]) >= 4
    for question in data["questions"]:
        assert {"id", "label", "question", "response"} <= set(question)
        AnswerResponse.model_validate(question["response"])


def test_bug_checks_are_real_bug_check_responses(data: dict[str, Any]) -> None:
    assert [c["id"] for c in data["bug_checks"]] == ["duplicate", "new", "injection"]
    checks = [*data["bug_checks"], {"id": "file_anyway", **data["file_anyway"]}]
    for check in checks:
        BugCheckRequest.model_validate(check["request"])
        response = BugCheckResponse.model_validate(check["response"])
        if response.proposed_action is not None:
            action = response.proposed_action
            assert (action.action_id, action.confirm_token, action.expires_at) == (None, None, None)
    assert data["file_anyway"]["request"].get("file_anyway") is True
    assert data["file_anyway"]["response"]["proposed_action"] is not None


def test_filed_issue_example_is_a_sandbox_issue(data: dict[str, Any]) -> None:
    url = data["filed_issue_example"]["url"]
    assert url.startswith(SANDBOX_ISSUES_PREFIX)
    parts = urlsplit(url)
    assert re.fullmatch(r"/raksharakyan/starfall-outpost-issues/issues/[1-9]\d*", parts.path)
    assert not parts.query and not parts.fragment


def test_voice_transcript_roles(data: dict[str, Any]) -> None:
    roles = {line["role"] for line in data["voice_transcript"]}
    assert roles <= {"agent", "user"}
    assert all(isinstance(line["text"], str) and line["text"] for line in data["voice_transcript"])


def test_showcase_json_has_no_secrets(data: dict[str, Any]) -> None:
    text = json.dumps(data)
    for needle in (
        'confirm_token": "',
        "gsk_",
        "sk-ant-",
        "xoxb-",
        "ghp_",
        "github_pat_",
        "hooks.slack.com",
    ):
        assert needle not in text, needle


# --- node runner -------------------------------------------------------------------------

REQUIRED_JS_TESTS = [
    "exports the documented helpers",
    "the showcase URL is relative and same-origin",
    "filedIssueLink accepts only sandbox repo issue URLs",
    "transcriptLines strips agent audio tags, keeps ids and drops unknown roles",
    "the committed showcase.json renders every example",
    "hostile recorded data is rendered as text only, with one same-origin fetch",
    "a failed load shows the error and enables nothing",
    "an unknown schema is treated as a failed load",
]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_showcase_js_node_runner() -> None:
    assert NODE is not None
    proc = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [NODE, str(JS_RUNNER), str(SHOWCASE_JS), str(SHOWCASE_JSON)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        check=False,
    )
    results = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    failures = [f"{r['name']}: {r.get('error')}" for r in results if not r["ok"]]
    assert not failures, "\n".join(failures)
    assert proc.returncode == 0, proc.stderr
    assert len(results) >= 18, f"only {len(results)} JS tests ran; stderr: {proc.stderr}"
    passed = {r["name"] for r in results if r["ok"]}
    missing = [name for name in REQUIRED_JS_TESTS if name not in passed]
    assert not missing, missing
