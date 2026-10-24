"""Minimal tests for the static showcase: scripts/record_showcase.py (--dry-run only: fake
LLM, hashing embedder, in-memory Qdrant, no keys, no network) and scripts/build_site.py.
The full suite is written by qa-agent."""

import html
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

from studiodesk.actions.proposals import InMemoryProposalStore, NewProposal
from studiodesk.models.actions import IssueDraft
from studiodesk.showcase import Showcase

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
STATIC = REPO_ROOT / "src" / "studiodesk" / "static"
FAKE_GROQ_KEY = "gsk_" + "Fak3NotReal" * 3


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recorder() -> ModuleType:
    return load_script("record_showcase")


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return load_script("build_site")


class RecordingProposals(InMemoryProposalStore):
    """Proposal store that remembers every id and token it handed out."""

    def __init__(self) -> None:
        super().__init__(900, max_pending=100)
        self.created: list[NewProposal] = []

    def create(self, draft: IssueDraft) -> NewProposal:
        proposal = super().create(draft)
        self.created.append(proposal)
        return proposal


@pytest.fixture(scope="module")
def recording(
    recorder: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, RecordingProposals]:
    out = tmp_path_factory.mktemp("showcase") / "showcase.json"
    proposals = RecordingProposals()
    args = recorder.parse_args(["--dry-run", "--out", str(out)])
    assert recorder.run(args, environ={}, proposals=proposals) == 0
    return out, proposals


def test_dry_run_writes_schema_valid_json(recording: tuple[Path, RecordingProposals]) -> None:
    out, _ = recording
    document = json.loads(out.read_text(encoding="utf-8"))
    showcase = Showcase.model_validate(document)
    assert document["schema"] == 1
    assert document["recorded_at"].endswith("Z")
    assert [q.id for q in showcase.questions] == [
        "cryo-save",
        "rover-crashes",
        "cross-save",
        "patch-1-3-1",
        "off-topic",
    ]
    assert [b.id for b in showcase.bug_checks] == ["duplicate", "new", "injection"]
    assert showcase.file_anyway.request.file_anyway is True
    assert showcase.voice_transcript[0].role == "agent"
    assert document["models"]["llm"] == "offline/offline-fake-llm"


def test_dry_run_output_has_no_tokens_or_action_ids(
    recording: tuple[Path, RecordingProposals],
) -> None:
    out, proposals = recording
    text = out.read_text(encoding="utf-8")
    assert proposals.created, "the file_anyway re-run should have proposed an issue"
    for proposal in proposals.created:
        assert proposal.confirm_token not in text
        assert proposal.action_id not in text
    document = json.loads(text)
    responses = [b["response"] for b in document["bug_checks"]]
    responses.append(document["file_anyway"]["response"])
    actions = [r["proposed_action"] for r in responses if r["proposed_action"] is not None]
    assert actions
    for action in actions:
        assert action["action_id"] is None
        assert action["confirm_token"] is None
        assert action["expires_at"] is None
        assert action["preview"]["title"]


def test_dry_run_must_not_overwrite_the_real_recording(recorder: ModuleType) -> None:
    with pytest.raises(SystemExit):
        recorder.parse_args(["--dry-run"])
    with pytest.raises(SystemExit):
        recorder.parse_args(["--dry-run", "--out", str(recorder.DEFAULT_OUT)])


def test_secret_in_recording_aborts_naming_field_not_value(
    recorder: ModuleType, tmp_path: Path
) -> None:
    out = tmp_path / "showcase.json"
    args = recorder.parse_args(["--dry-run", "--out", str(out)])
    secret = "hover-rover"  # noqa: S105 (appears in a question; stands in for a leaked value)
    with pytest.raises(recorder.SecretFoundError) as excinfo:
        recorder.run(args, environ={"SHOWCASE_TEST_TOKEN": secret})
    message = str(excinfo.value)
    assert "$.questions[1].question" in message
    assert secret not in message
    assert not out.exists()


def test_recorder_only_calls_ask_and_bug_check(recorder: ModuleType) -> None:
    for path in ("/actions/abc/confirm", "/actions/abc/cancel", "/voice/session"):
        with pytest.raises(recorder.RecorderError):
            recorder.post_json(object(), path, {}, attempts=1, retry_wait_s=0, sleep=lambda s: None)


def test_github_and_slack_stand_ins_never_send(recorder: ModuleType) -> None:
    draft = IssueDraft(title="t", body="b")
    with pytest.raises(recorder.RecorderError):
        recorder.NoFilingGitHub("owner/repo").create(draft)
    slack = recorder.NoSlack()
    assert slack.enabled is False
    with pytest.raises(recorder.RecorderError):
        slack.notify_issue_created(object(), draft)


def test_throttled_llm_spaces_calls(recorder: ModuleType) -> None:
    now = [0.0]
    sleeps: list[float] = []

    class Inner:
        def structured(self, system: str, user: str, schema: type[BaseModel]) -> Any:
            now[0] += 1.0  # each call takes 1 s
            return None

        def close(self) -> None:
            pass

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    llm = recorder.ThrottledLLM(Inner(), 12.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        llm.structured("s", "u", BaseModel)
    assert sleeps == [12.0, 12.0]


def test_examples_match_the_main_page(recorder: ModuleType) -> None:
    index = (STATIC / "index.html").read_text(encoding="utf-8")
    chips = [html.unescape(q) for q in re.findall(r'data-question="([^"]*)"', index)]
    assert chips == [chip.question for chip in recorder.QUESTIONS]
    # Join the JS string concatenations ("..." + "...") so each value is one literal.
    home = re.sub(r'"\s*\+\s*"', "", (STATIC / "home.js").read_text(encoding="utf-8"))
    for bug_id, example in recorder.BUG_EXAMPLES.items():
        assert re.search(rf"\b{bug_id}: Object\.freeze", home)
        for key, value in example.items():
            assert f"{key}: {json.dumps(value)}" in re.sub(r":\s+", ": ", home), (bug_id, key)


# ---------------------------------------------------------------------------
# build_site.py
# ---------------------------------------------------------------------------


def make_site(root: Path, document: dict[str, Any]) -> Path:
    site = root / "site"
    (site / "data").mkdir(parents=True)
    (site / "index.html").write_text("<!doctype html><title>StudioDesk</title>\n")
    (site / "data" / "showcase.json").write_text(json.dumps(document), encoding="utf-8")
    return site


def recorded(recording: tuple[Path, RecordingProposals]) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(recording[0].read_text(encoding="utf-8"))
    return document


def build_args(site: Path, out: Path, tmp_path: Path) -> list[str]:
    return ["--site-dir", str(site), "--out", str(out), "--env-file", str(tmp_path / "no.env")]


def test_build_site_copies_site_and_styles(
    builder: ModuleType, recording: tuple[Path, RecordingProposals], tmp_path: Path
) -> None:
    site = make_site(tmp_path, recorded(recording))
    out = tmp_path / "_site"
    assert builder.main(build_args(site, out, tmp_path)) == 0
    assert (out / "index.html").is_file()
    assert (out / "data" / "showcase.json").is_file()
    styles = (STATIC / "styles.css").read_bytes()
    assert (out / "assets" / "styles.css").read_bytes() == styles


def test_build_site_rejects_a_groq_token(
    builder: ModuleType,
    recording: tuple[Path, RecordingProposals],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = recorded(recording)
    document["questions"][0]["response"]["answer"] = f"Use {FAKE_GROQ_KEY} to log in."
    site = make_site(tmp_path, document)
    out = tmp_path / "_site"
    assert builder.main(build_args(site, out, tmp_path)) == 2
    err = capsys.readouterr().err
    assert "$.questions[0].response.answer looks like a Groq key" in err
    assert FAKE_GROQ_KEY not in err
    assert not out.exists()


def test_build_site_rejects_an_unredacted_confirm_token(
    builder: ModuleType,
    recording: tuple[Path, RecordingProposals],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = recorded(recording)
    action = document["file_anyway"]["response"]["proposed_action"]
    action["confirm_token"] = "A" * 43
    site = make_site(tmp_path, document)
    assert builder.main(build_args(site, tmp_path / "_site", tmp_path)) == 2
    assert "schema check failed" in capsys.readouterr().err


def test_build_site_refuses_symlinks(
    builder: ModuleType, recording: tuple[Path, RecordingProposals], tmp_path: Path
) -> None:
    site = make_site(tmp_path, recorded(recording))
    (site / "link").symlink_to(REPO_ROOT / "README.md")
    assert builder.main(build_args(site, tmp_path / "_site", tmp_path)) == 2
    shutil.rmtree(site)
