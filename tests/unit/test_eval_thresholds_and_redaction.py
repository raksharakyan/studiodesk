"""Threshold floors, report redaction and command sanitising for the eval harness."""

import json
from pathlib import Path

import pytest
from evals.checks import REDACTED, redact_secrets
from evals.report import write_report
from evals.run import REPO_ROOT, sanitise_command
from evals.suites import SuiteResult
from evals.thresholds import (
    THRESHOLDS_PATH,
    ThresholdCheck,
    check_thresholds,
    load_thresholds,
    metric_value,
)

# ------------------------------------------------------------------------ thresholds


def test_repo_thresholds_file_has_the_agreed_floors() -> None:
    assert load_thresholds(THRESHOLDS_PATH) == {
        "retrieval.hit@5": 0.9,
        "answer.citation_validity": 1.0,
        "injection.pass_rate": 1.0,
        "duplicates.heldout.f1": 0.85,
        "routing.all.component_accuracy": 0.45,
    }


def test_missing_thresholds_file_means_no_floors(tmp_path: Path) -> None:
    assert load_thresholds(tmp_path / "absent.json") == {}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "invalid JSON"),
        ('{"floors": [1]}', "'floors' mapping"),
        ('{"floors": {"retrieval.hit@5": 1.5}}', r"\[0, 1\]"),
        ('{"floors": {"retrieval.hit@5": true}}', r"\[0, 1\]"),
        ('{"floors": {"retrieval.hit@5": "0.9"}}', r"\[0, 1\]"),
        ('{"floors": {"hit@5": 0.9}}', "<suite>"),
    ],
)
def test_invalid_thresholds_are_rejected(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "t.json"
    path.write_text(content)

    with pytest.raises(ValueError, match=message):
        load_thresholds(path)


def test_metric_value_walks_nested_metrics() -> None:
    metrics = {"heldout": {"f1": 0.9, "n": None}, "hit@5": 1, "flag": True}

    assert metric_value(metrics, "heldout.f1") == 0.9
    assert metric_value(metrics, "hit@5") == 1.0
    assert metric_value(metrics, "heldout.n") is None
    assert metric_value(metrics, "heldout.missing") is None
    assert metric_value(metrics, "flag") is None


def _result(name: str, metrics: dict[str, object]) -> SuiteResult:
    return SuiteResult(name=name, metrics=metrics, cases=[], failures=[])


def test_check_thresholds_only_for_suites_that_ran() -> None:
    results = {
        "retrieval": _result("retrieval", {"hit@5": 0.88}),
        "duplicates": _result("duplicates", {"heldout": {"f1": None}}),
    }
    floors = {"retrieval.hit@5": 0.9, "duplicates.heldout.f1": 0.85, "routing.all.x": 0.4}

    checks = check_thresholds(results, floors)

    assert checks == [
        ThresholdCheck("retrieval.hit@5", 0.9, 0.88),
        ThresholdCheck("duplicates.heldout.f1", 0.85, None),
    ]
    assert [c.ok for c in checks] == [False, False]  # None = no completed cases
    assert ThresholdCheck("a.b", 0.9, 0.9).ok


# ------------------------------------------------------------------------- redaction


def test_redacts_configured_secret_raw_and_json_escaped() -> None:
    marker = 'my"configured\\value-123'
    text = f"raw {marker} json {json.dumps(marker)[1:-1]}"

    cleaned, count = redact_secrets(text, [marker])

    assert marker not in cleaned
    assert json.dumps(marker)[1:-1] not in cleaned
    assert cleaned == f"raw {REDACTED} json {REDACTED}"
    assert count == 2


@pytest.mark.parametrize(
    "token",
    [
        "gsk" + "_abcdefgh12345678",
        "github" + "_pat_11ABCDEFG_abcdefgh",
        "ghp" + "_abcdefgh12345678",
        "sk" + "-ant-api03-abcdefghijkl",
        "xoxb" + "-1234567890-abcdef",
        "https://hooks.slack.com/" + "services/T000/B000/XXX",
    ],
)
def test_redacts_secret_shaped_tokens(token: str) -> None:
    cleaned, count = redact_secrets(f"leaked: {token} (end)")

    assert token not in cleaned
    assert REDACTED in cleaned
    assert count == 1


def test_redacts_api_key_assignment_value_and_is_idempotent() -> None:
    cleaned, count = redact_secrets("config api_key=abc123 and API-KEY: zzz")

    assert cleaned == f"config api_key={REDACTED} and API-KEY: {REDACTED}"
    assert count == 2
    assert redact_secrets(cleaned) == (cleaned, 0)


def test_clean_text_is_unchanged() -> None:
    text = "Fixed in 1.0.1 [BUG-0001]; marker patterns: ['pattern:gsk_[A-Za-z0']"
    assert redact_secrets(text, ["not-present-value"]) == (text, 0)


def test_write_report_redacts_before_writing(tmp_path: Path) -> None:
    marker = "configured-marker-value-xyz"
    result = SuiteResult(
        name="retrieval",
        metrics={
            "cases": 1,
            "completed": 1,
            "errored": 0,
            "hit@5": 1.0,
            "mrr@5": 1.0,
            "recall@5": 1.0,
            "hit@1": 1.0,
        },
        cases=[{"id": "c1", "hit": 1.0, "rr": 1.0, "recall": 1.0, "retrieved": [marker]}],
        failures=[("c1", f"model said {marker} and gsk" + "_abcdefgh12345678")],
    )
    meta = {
        "offline": False,
        "date": "d",
        "git_sha": "s",
        "command": "c",
        "answerer": "a",
        "judge": "j",
        "embedding_model": "e",
        "embedding_revision": "r",
        "qdrant": "q",
        "agent_settings": {},
        "case_files": {},
        "exit_code": 0,
    }

    md_path, json_path, redactions = write_report(
        tmp_path, "latest", meta, {"retrieval": result}, {}, (), [marker]
    )

    for path in (md_path, json_path):
        content = path.read_text()
        assert marker not in content
        assert "gsk" + "_abcdefgh" not in content
        assert REDACTED in content
    json.loads(json_path.read_text())  # still valid JSON
    assert redactions >= 4


# ---------------------------------------------------------------------- command line


def test_command_paths_are_sanitised(tmp_path: Path) -> None:
    command = sanitise_command(
        [
            "--offline",
            "--output-dir",
            str(tmp_path / "reports"),
            f"--cache-dir={tmp_path / 'cache'}",
            "--cases-dir",
            str(REPO_ROOT / "evals" / "cases"),
            "--suite",
            "answer",
        ]
    )

    assert str(tmp_path) not in command
    assert str(Path.home()) not in command
    assert command == (
        "uv run python -m evals.run --offline --output-dir <abs>/reports "
        "--cache-dir=<abs>/cache --cases-dir evals/cases --suite answer"
    )
