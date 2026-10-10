"""The non-offline harness path (cache, rate limiter, real Settings) with a fake LLM.

`build_llm` and the embedder are replaced so no network, key or model download is
needed; everything else (preflight, CachedLLM, RateLimiter, report, exit codes) is the
code a live run executes.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from evals import run as eval_run
from evals.fakes import HashingEmbedder, OfflineLLM
from evals.judge import JudgedClaim, JudgeVerdict
from pydantic import BaseModel

from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.llm import LLMClient, LLMUnavailable
from studiodesk.models.bugs import DuplicateJudgements

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_KEY = "fake-provider-key-for-tests-0001"


class LeakyJudgeLLM(OfflineLLM):
    """Offline answers, but the judge echoes the configured key in an unsupported claim."""

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        if schema is JudgeVerdict:
            verdict = JudgeVerdict(claims=[JudgedClaim(text=f"key is {FAKE_KEY}", supported=False)])
            assert isinstance(verdict, schema)
            return verdict
        return super().structured(system, user_content, schema)


class FailingDuplicatesLLM(OfflineLLM):
    """Offline answers, but every duplicate judgement fails like a rate-limited provider."""

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        if schema is DuplicateJudgements:
            raise LLMUnavailable("groq call failed: RateLimitError(429)", retryable=True)
        return super().structured(system, user_content, schema)


@pytest.fixture
def live_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Empty cwd (no .env), a fake key, a fake LLM factory and the hashing embedder."""
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_PROVIDER", "QDRANT_URL", "APP_ENV", "ANTHROPIC_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    state: dict[str, Any] = {"llm_class": OfflineLLM, "built": []}

    def fake_build_llm(settings: Settings) -> LLMClient:
        state["built"].append(settings.resolved_llm_model)
        state["max_retries"] = settings.llm_max_retries
        llm: LLMClient = state["llm_class"]()
        return llm

    def fake_embedder(offline: bool, settings: Settings) -> tuple[Embedder, str, str]:
        return HashingEmbedder(), "fake-embedder", "fake-revision"

    monkeypatch.setattr(eval_run, "build_llm", fake_build_llm)
    monkeypatch.setattr(eval_run, "build_embedder", fake_embedder)
    return state


def _main(tmp_path: Path, *args: str) -> int:
    return eval_run.main(
        ["--output-dir", str(tmp_path / "out"), "--cache-dir", str(tmp_path / "cache"), *args]
    )


def _report(tmp_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((tmp_path / "out" / "latest.json").read_text())
    return report


def test_live_path_uses_cache_and_limiter(tmp_path: Path, live_env: dict[str, Any]) -> None:
    assert _main(tmp_path, "--suite", "answer", "--max-tpm", "1000000") == 0

    first = _report(tmp_path)
    assert live_env["built"] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    assert live_env["max_retries"] == 3  # batch retry policy
    assert first["meta"]["answerer"] == "groq:openai/gpt-oss-120b"
    assert first["meta"]["judge"] == "groq:qwen/qwen3.8-27b"
    assert first["cost"]["max_tpm_per_model"] == 1_000_000
    assert first["cost"]["max_rpm_per_model"] == 20
    assert first["cost"]["llm_calls"] > 0
    assert list((tmp_path / "cache").glob("*.json"))
    assert not (tmp_path / "out" / "offline.md").exists()

    assert _main(tmp_path, "--suite", "answer", "--max-tpm", "1000000") == 0

    second = _report(tmp_path)
    assert second["cost"]["llm_calls"] == 0
    total = first["cost"]["llm_calls"] + first["cost"]["cache_hits"]
    assert second["cost"]["cache_hits"] == total
    for entry in (tmp_path / "cache").glob("*.json"):
        assert FAKE_KEY not in entry.read_text()


def test_errored_cases_are_separate_and_exit_non_zero(
    tmp_path: Path, live_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    live_env["llm_class"] = FailingDuplicatesLLM

    code = _main(tmp_path, "--suite", "duplicates", "--max-tpm", "1000000", "--max-rpm", "1000")

    assert code == 1
    assert "errored" in capsys.readouterr().err
    report = _report(tmp_path)
    suite = report["suites"]["duplicates"]
    errored_ids = {e["id"] for e in suite["errored"]}
    assert errored_ids
    assert errored_ids.isdisjoint({f["id"] for f in suite["failures"]})
    overall = suite["metrics"]["overall"]
    assert overall["errored"] == len(errored_ids)
    assert overall["completed"] == overall["cases"] - len(errored_ids)
    assert report["summary"][0]["errored"] == len(errored_ids)
    assert report["meta"]["exit_code"] == 1
    md = (tmp_path / "out" / "latest.md").read_text()
    assert "## Errored cases" in md
    assert "LLMUnavailable" in md


def test_strict_fails_below_floor_only_when_requested(
    tmp_path: Path, live_env: dict[str, Any]
) -> None:
    # The hashing embedder scores far below the retrieval floor.
    assert _main(tmp_path, "--suite", "retrieval") == 0
    assert _main(tmp_path, "--suite", "retrieval", "--strict") == 1
    report = _report(tmp_path)
    assert report["thresholds"] == [
        {
            "metric": "retrieval.hit@5",
            "floor": 0.9,
            "value": report["suites"]["retrieval"]["metrics"]["hit@5"],
            "ok": False,
        }
    ]


def test_strict_passes_when_floors_are_met(tmp_path: Path, live_env: dict[str, Any]) -> None:
    floors = tmp_path / "floors.json"
    floors.write_text('{"floors": {"retrieval.hit@5": 0.0}}')

    assert _main(tmp_path, "--suite", "retrieval", "--strict", "--thresholds", str(floors)) == 0


def test_invalid_thresholds_file_is_a_config_error(
    tmp_path: Path, live_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    floors = tmp_path / "floors.json"
    floors.write_text("{broken")

    assert _main(tmp_path, "--suite", "retrieval", "--thresholds", str(floors)) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_strict_is_rejected_offline() -> None:
    with pytest.raises(SystemExit) as exc:
        eval_run.main(["--offline", "--strict"])
    assert exc.value.code == 2


def test_report_redacts_a_key_echoed_by_the_model(
    tmp_path: Path, live_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    live_env["llm_class"] = LeakyJudgeLLM

    assert _main(tmp_path, "--suite", "answer", "--max-tpm", "1000000") == 0

    out = capsys.readouterr()
    assert "redacted" in out.err
    for name in ("latest.md", "latest.json"):
        content = (tmp_path / "out" / name).read_text()
        assert FAKE_KEY not in content
        assert "[REDACTED]" in content
    assert FAKE_KEY not in out.out + out.err


def test_report_command_has_no_local_paths(tmp_path: Path, live_env: dict[str, Any]) -> None:
    assert _main(tmp_path, "--suite", "retrieval") == 0

    command = _report(tmp_path)["meta"]["command"]
    assert str(tmp_path) not in command
    assert "<abs>/out" in command


def test_harness_import_loads_no_action_modules() -> None:
    code = (
        "import sys\n"
        "import evals.run, evals.suites, evals.report, evals.fakes\n"
        "from evals.suites import ACTION_MODULES, action_modules_loaded\n"
        "print(','.join(action_modules_loaded()))\n"
    )
    result = subprocess.run(  # noqa: S603 (fixed argv, current interpreter)
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    assert result.stdout.strip() == ""
