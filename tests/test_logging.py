"""JSON logging and secret redaction."""

import json
import logging

import pytest
from pydantic import SecretStr

from studiodesk.config import Settings
from studiodesk.logging import REDACTED, SecretRedactingFilter, configure_logging
from studiodesk.main import create_app

SECRET = "sk-ant-THIS-MUST-NOT-LEAK"  # noqa: S105 - fake test secret
SECRET_STR = SecretStr(SECRET)
TOKEN_STR = SecretStr("ghp_TOKEN1")


def _lines(captured: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


def test_secret_in_message_is_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """A secret interpolated into a log message is replaced by the redaction marker."""
    settings = Settings(_env_file=None, anthropic_api_key=SECRET_STR, github_token=TOKEN_STR)
    configure_logging(settings)

    logging.getLogger("studiodesk.test").warning("key=%s token=%s", SECRET, "ghp_TOKEN1")

    out = capsys.readouterr().out
    assert SECRET not in out
    assert "ghp_TOKEN1" not in out
    record = _lines(out)[-1]
    assert record["message"] == f"key={REDACTED} token={REDACTED}"
    assert record["level"] == "WARNING"
    assert record["logger"] == "studiodesk.test"


def test_every_secret_field_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """All SecretStr fields on Settings are redacted, not just the LLM key."""
    secrets = {
        "anthropic_api_key": "secret-anthropic-1",
        "qdrant_api_key": "secret-qdrant-2",
        "github_token": "secret-github-3",
        "slack_webhook_url": "https://hooks.slack.com/services/T000/B000/fakesecret4",
        "elevenlabs_api_key": "secret-eleven-5",
    }
    configure_logging(Settings(_env_file=None, **secrets))  # type: ignore[arg-type]

    logging.getLogger("studiodesk.test").error(" | ".join(secrets.values()))

    out = capsys.readouterr().out
    for value in secrets.values():
        assert value not in out


def test_logging_emits_json_lines_with_extra(capsys: pytest.CaptureFixture[str]) -> None:
    """Records are single-line JSON including `extra=` fields."""
    configure_logging(Settings(_env_file=None, log_level="DEBUG"))

    logging.getLogger("studiodesk.test").debug("hello", extra={"bug_id": "BUG-0001"})

    record = _lines(capsys.readouterr().out)[-1]
    assert record["message"] == "hello"
    assert record["bug_id"] == "BUG-0001"
    assert record["level"] == "DEBUG"
    assert "ts" in record


def test_log_level_applied(capsys: pytest.CaptureFixture[str]) -> None:
    """Records below the configured level are dropped."""
    configure_logging(Settings(_env_file=None, log_level="WARNING"))

    logging.getLogger("studiodesk.test").info("should not appear")

    assert "should not appear" not in capsys.readouterr().out


def test_configure_logging_is_idempotent() -> None:
    """Re-configuring replaces the StudioDesk handler instead of stacking another."""
    configure_logging(Settings(_env_file=None))
    configure_logging(Settings(_env_file=None))

    names = [h.get_name() for h in logging.getLogger().handlers]
    assert names.count("studiodesk-json") == 1


def test_exception_logged_with_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    """logger.exception includes formatted exc_info in the JSON record."""
    configure_logging(Settings(_env_file=None))
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("studiodesk.test").exception("failed")

    record = _lines(capsys.readouterr().out)[-1]
    assert "RuntimeError: boom" in str(record["exc_info"])


def test_create_app_logs_without_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    """Building the app logs startup info but never configured secrets."""
    create_app(Settings(_env_file=None, app_env="test", anthropic_api_key=SECRET_STR))

    out = capsys.readouterr().out
    assert "app created" in out
    assert SECRET not in out


def test_filter_without_secrets_leaves_record_untouched() -> None:
    """With no secrets configured the filter passes records through unchanged."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "a %s", ("b",), None)

    assert SecretRedactingFilter(["", ""]).filter(record) is True
    assert record.msg == "a %s"
    assert record.args == ("b",)


def test_secret_in_extra_field_is_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """A secret passed via `extra=` must not reach the log output either."""
    configure_logging(Settings(_env_file=None, anthropic_api_key=SECRET_STR))

    logging.getLogger("studiodesk.test").info("calling llm", extra={"auth": SECRET})

    assert SECRET not in capsys.readouterr().out


def test_secret_in_exception_is_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """A secret embedded in an exception message must not reach the log output."""
    configure_logging(Settings(_env_file=None, anthropic_api_key=SECRET_STR))
    try:
        raise RuntimeError(f"auth failed for key {SECRET}")
    except RuntimeError:
        logging.getLogger("studiodesk.test").exception("llm call failed")

    assert SECRET not in capsys.readouterr().out


def test_stack_info_included_and_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """stack_info is emitted and passes through redaction."""
    configure_logging(Settings(_env_file=None, anthropic_api_key=SECRET_STR))

    logging.getLogger("studiodesk.test").info("trace", stack_info=True, extra={"k": SECRET})

    out = capsys.readouterr().out
    record = _lines(out)[-1]
    assert "Stack (most recent call last)" in str(record["stack_info"])
    assert SECRET not in out


def test_secret_with_json_escaped_chars_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """Secrets containing quotes/backslashes/non-ASCII are redacted after JSON escaping."""
    tricky = 'p"a\\ssé-word'
    configure_logging(Settings(_env_file=None, qdrant_api_key=SecretStr(tricky)))

    logging.getLogger("studiodesk.test").info("x", extra={"cred": tricky})

    out = capsys.readouterr().out
    assert tricky not in out
    assert json.dumps(tricky)[1:-1] not in out
    assert _lines(out)[-1]["cred"] == REDACTED


def test_uvicorn_loggers_routed_through_redacting_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """uvicorn's own handlers are dropped so its records are JSON and redacted too."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).addHandler(logging.StreamHandler())
    configure_logging(Settings(_env_file=None, anthropic_api_key=SECRET_STR))

    logging.getLogger("uvicorn.error").warning("bad header %s", SECRET)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        server_logger = logging.getLogger(name)
        assert server_logger.handlers == []
        assert server_logger.propagate is True
    out = capsys.readouterr().out
    assert SECRET not in out
    assert _lines(out)[-1]["logger"] == "uvicorn.error"


@pytest.mark.xfail(strict=True, reason="httpx logger level pending")
@pytest.mark.parametrize("level", ["DEBUG", "INFO"])
def test_httpx_logger_quietened(level: str) -> None:
    """httpx logs full request URLs at INFO; configure_logging should raise it to WARNING."""
    httpx_logger = logging.getLogger("httpx")
    saved = httpx_logger.level
    try:
        configure_logging(Settings(_env_file=None, log_level=level))  # type: ignore[arg-type]
        assert httpx_logger.getEffectiveLevel() >= logging.WARNING
    finally:
        httpx_logger.setLevel(saved)
