"""Static checks on the M6 main page (index.html) and its script (home.js): no HTML or code
sinks, no inline script/style/handlers, the pinned SDK with the same SRI as voice.html, no
storage or console in home.js, accessibility basics, and that every element id the scripts
look up exists on the pages that load them."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from studiodesk.api.pages import ELEVENLABS_SDK_URL

STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "studiodesk" / "static"
INDEX_HTML = STATIC_DIR / "index.html"
VOICE_HTML = STATIC_DIR / "voice.html"
HOME_JS = STATIC_DIR / "home.js"
VOICE_JS = STATIC_DIR / "voice.js"


class Page(HTMLParser):
    """Collects the tags, attributes and text structure the checks need."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.inline_scripts: list[str] = []
        self.style_blocks = 0
        self._in_script = False
        self._in_style = False

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

    def by_id(self, element_id: str) -> tuple[str, dict[str, str | None]]:
        [match] = [(t, a) for t, a in self.tags if a.get("id") == element_id]
        return match


def parse(path: Path) -> Page:
    page = Page()
    page.feed(path.read_text(encoding="utf-8"))
    return page


@pytest.fixture(scope="module")
def index() -> Page:
    return parse(INDEX_HTML)


@pytest.fixture(scope="module")
def home_js() -> str:
    return HOME_JS.read_text(encoding="utf-8")


def code_only(js: str) -> str:
    """JS without // line comments and /* */ blocks, so prose is not matched."""
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", js)


# --- home.js -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sink",
    [
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        'setTimeout("',
        "createContextualFragment",
        "DOMParser",
    ],
)
def test_home_js_has_no_html_or_code_sinks(home_js: str, sink: str) -> None:
    assert sink not in code_only(home_js)


@pytest.mark.parametrize(
    "api", ["localStorage", "sessionStorage", "indexedDB", "document.cookie", "console."]
)
def test_home_js_has_no_storage_or_console(home_js: str, api: str) -> None:
    assert api not in code_only(home_js)


def test_home_js_sets_no_handlers_or_attributes_from_data(home_js: str) -> None:
    code = code_only(home_js)
    assert not re.search(r"\.on[a-z]+\s*=", code)
    assert "setAttribute" not in code
    # Only .href (from validated URLs) and .textContent / .value / .hidden are assigned.
    assigned = set(re.findall(r"\.(\w+)\s*=(?!=)", code))
    assert assigned <= {"href", "textContent", "value", "hidden"}, assigned


def test_home_js_fetches_only_demo_info_same_origin(home_js: str) -> None:
    code = code_only(home_js)
    assert re.findall(r"fetch\(\s*([^,)]+)", code) == ['"/api/demo-info"']
    assert 'redirect: "error"' in code and 'credentials: "same-origin"' in code


def test_home_js_imports_only_voice_js(home_js: str) -> None:
    assert re.findall(r'^import .* from "([^"]+)";', home_js, re.M) == ["./voice.js"]


# --- index.html: script, style and handler hygiene -----------------------------------


def test_index_has_no_inline_scripts_styles_or_handlers(index: Page) -> None:
    assert index.inline_scripts == []
    assert index.style_blocks == 0
    for tag, attrs in index.tags:
        for name in attrs:
            assert not name.lower().startswith("on"), (tag, name)
            assert name.lower() != "style", tag
        for value in attrs.values():
            assert not (value or "").strip().lower().startswith("javascript:"), (tag, attrs)


def test_index_scripts_are_the_pinned_sdk_and_home_js(index: Page) -> None:
    scripts = index.find("script")
    external = [s for s in scripts if (s.get("src") or "").startswith("http")]
    local = [s for s in scripts if not (s.get("src") or "").startswith("http")]
    assert [s["src"] for s in external] == [ELEVENLABS_SDK_URL]
    assert [s["src"] for s in local] == ["/static/home.js"]
    assert local[0].get("type") == "module"
    [sdk] = external
    assert sdk.get("crossorigin") == "anonymous"
    [voice_sdk] = [
        s for s in parse(VOICE_HTML).find("script") if s.get("src") == ELEVENLABS_SDK_URL
    ]
    assert sdk.get("integrity") == voice_sdk.get("integrity")
    assert re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", sdk.get("integrity") or "")


def test_index_links_are_local_or_https(index: Page) -> None:
    for attrs in index.find("link"):
        href = attrs.get("href") or ""
        assert href.startswith(("/static/", "data:")), href
    for attrs in index.find("a"):
        href = attrs.get("href") or ""
        assert href.startswith(("#", "/", "https://github.com/")), href
        if attrs.get("target") == "_blank":
            assert "noopener" in (attrs.get("rel") or ""), href


# --- index.html: accessibility -------------------------------------------------------


def test_index_has_lang_and_single_h1(index: Page) -> None:
    [html] = index.find("html")
    assert html.get("lang") == "en"
    assert len(index.find("h1")) == 1


def test_index_skip_link_targets_main(index: Page) -> None:
    anchors = index.find("a")
    assert anchors[0].get("href") == "#main", "the skip link must be the first link"
    assert "skip-link" in (anchors[0].get("class") or "")
    tag, _ = index.by_id("main")
    assert tag == "main"


def test_index_ids_are_unique(index: Page) -> None:
    ids = index.ids()
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, duplicates


def test_every_form_control_has_a_label(index: Page) -> None:
    labelled = {attrs.get("for") for attrs in index.find("label")}
    controls = [
        (tag, attrs)
        for tag, attrs in index.tags
        if tag in ("input", "textarea", "select") and attrs.get("type") != "hidden"
    ]
    assert controls
    for tag, attrs in controls:
        assert attrs.get("id") in labelled or attrs.get("aria-label"), (tag, attrs.get("id"))


def test_buttons_declare_their_type(index: Page) -> None:
    for attrs in index.find("button"):
        assert attrs.get("type") in ("button", "submit"), attrs


@pytest.mark.parametrize(
    "element_id", ["ask-status", "bug-status", "action-status", "voice-status", "voice-hint"]
)
def test_status_regions_are_polite(index: Page, element_id: str) -> None:
    _, attrs = index.by_id(element_id)
    assert attrs.get("role") == "status"
    assert attrs.get("aria-live") == "polite"


def test_voice_hint_is_empty_polite_status(index: Page) -> None:
    tag, attrs = index.by_id("voice-hint")
    assert tag == "p" and attrs.get("role") == "status" and attrs.get("aria-live") == "polite"


@pytest.mark.parametrize("element_id", ["voice-error", "proposal-changed"])
def test_alerts_use_role_alert(index: Page, element_id: str) -> None:
    _, attrs = index.by_id(element_id)
    assert attrs.get("role") == "alert"


def test_images_have_alt(index: Page) -> None:
    for attrs in index.find("img"):
        assert "alt" in attrs, attrs


# --- scripts and pages agree on element ids ------------------------------------------


def looked_up_ids(js: str) -> set[str]:
    """Ids passed to the scripts' `$("...")` helper (required elements)."""
    return set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', code_only(js)))


def test_ids_used_by_voice_js_exist_on_both_pages() -> None:
    needed = looked_up_ids(VOICE_JS.read_text(encoding="utf-8"))
    assert needed
    for page_path in (INDEX_HTML, VOICE_HTML):
        missing = needed - set(parse(page_path).ids())
        assert not missing, f"{page_path.name} lacks {sorted(missing)}"


def test_ids_used_by_home_js_exist_on_the_main_page(home_js: str) -> None:
    needed = looked_up_ids(home_js)
    assert needed
    missing = needed - set(parse(INDEX_HTML).ids())
    assert not missing, sorted(missing)


def test_main_page_has_the_chips_and_voice_section(index: Page) -> None:
    ids = set(index.ids())
    assert {"try", "voice", "file-anyway", "file-anyway-btn"} <= ids
    question_chips = [a for t, a in index.tags if t == "button" and a.get("data-question")]
    bug_chips = {a.get("data-bug") for t, a in index.tags if t == "button" and a.get("data-bug")}
    assert question_chips
    assert bug_chips == {"duplicate", "new", "injection"}
