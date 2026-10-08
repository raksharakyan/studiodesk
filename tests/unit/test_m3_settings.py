"""M3 security settings: actions kill switch, LLM cost bounds and `.env.example` defaults.

Settings are always built with `_env_file=None` (or the checked-in `.env.example`) and
the relevant variables removed from the process environment, so no real `.env` is read.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from studiodesk.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
PROD = {"app_env": "prod", "qdrant_url": "https://cluster.example.com"}
VARS = (
    "APP_ENV",
    "ACTIONS_ENABLED",
    "ACTIONS_MAX_PER_DAY",
    "LLM_TIMEOUT_S",
    "LLM_REQUEST_DEADLINE_S",
    "LLM_MAX_CONCURRENCY",
    "LLM_CONCURRENCY_WAIT_S",
    "LLM_BUSY_RETRY_AFTER_S",
    "SLACK_WEBHOOK_URL",
    "QDRANT_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VARS:
        monkeypatch.delenv(name, raising=False)


def test_prod_disables_actions_by_default() -> None:
    assert Settings(_env_file=None, **PROD).actions_enabled is False


def test_prod_with_explicit_true_enables_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None, actions_enabled=True, **PROD).actions_enabled is True
    monkeypatch.setenv("ACTIONS_ENABLED", "true")
    assert Settings(_env_file=None, **PROD).actions_enabled is True


def test_prod_with_explicit_false_stays_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTIONS_ENABLED", "false")
    assert Settings(_env_file=None, **PROD).actions_enabled is False


@pytest.mark.parametrize("env", ["dev", "test"])
def test_dev_and_test_enable_actions_by_default(env: str) -> None:
    assert Settings(_env_file=None, app_env=env).actions_enabled is True
    assert Settings(_env_file=None, app_env=env, actions_enabled=False).actions_enabled is False


def test_new_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.llm_timeout_s == 30
    assert s.llm_request_deadline_s == 75
    assert s.llm_max_concurrency == 4
    assert s.llm_concurrency_wait_s == 2
    assert s.llm_busy_retry_after_s == 5
    assert s.actions_max_per_day == 20


@pytest.mark.parametrize(
    "override",
    [
        {"llm_max_concurrency": 0},
        {"llm_max_concurrency": 65},
        {"llm_request_deadline_s": 0},
        {"llm_concurrency_wait_s": -1},
        {"llm_busy_retry_after_s": 0},
        {"actions_max_per_day": 0},
    ],
)
def test_new_bounds_are_validated(override: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)


def test_env_example_slack_webhook_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GITHUB_REPO"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.slack_webhook_url is None
    assert settings.actions_max_per_day == 20
    assert settings.llm_timeout_s == 30
    assert settings.llm_request_deadline_s == 75
    assert settings.llm_max_concurrency == 4
