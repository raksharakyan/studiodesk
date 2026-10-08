"""Minimal structured (JSON-lines) logging configured from Settings.

Secret values from Settings are redacted from every emitted line (message, `extra=` fields
and tracebacks) as a defence in depth; code should still never pass secrets to a logger.
"""

import json
import logging
import sys
from datetime import UTC, datetime

from pydantic import SecretStr

from studiodesk.config import Settings

REDACTED = "**********"
_HANDLER_NAME = "studiodesk-json"
# Loggers that servers configure with their own handlers; they are re-pointed at the root
# JSON handler so their output is redacted too.
_SERVER_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
# HTTP client and SDK loggers emit request URLs/details at INFO/DEBUG; keep them at WARNING.
_QUIET_LOGGERS = ("httpx", "httpcore", "httpx2", "anthropic")
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys() | {"message", "asctime"}
)


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects, redacting known secrets.

    Redaction runs on the final serialised line, so it covers the message, every `extra=`
    field and formatted tracebacks alike.
    """

    def __init__(self, secrets: list[str] | None = None) -> None:
        """Store the non-empty secret strings to redact from every emitted line."""
        super().__init__()
        self._secrets = _redaction_needles(secrets or [])

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record, including any `extra=` fields, to redacted JSON."""
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return self._redact(json.dumps(payload, default=str))

    def _redact(self, line: str) -> str:
        """Replace every occurrence of a secret (raw or JSON-escaped) in `line`."""
        for needle in self._secrets:
            line = line.replace(needle, REDACTED)
        return line


def _redaction_needles(secrets: list[str]) -> list[str]:
    """Return each secret plus its JSON-escaped form, longest first, without empties.

    `json.dumps` escapes quotes, backslashes, control and non-ASCII characters, so a secret
    containing any of them would not match its raw value in serialised output.
    """
    needles: set[str] = set()
    for secret in secrets:
        if secret:
            needles.add(secret)
            needles.add(json.dumps(secret)[1:-1])
    return sorted(needles, key=len, reverse=True)


class SecretRedactingFilter(logging.Filter):
    """Replace any known secret value appearing in a formatted log message."""

    def __init__(self, secrets: list[str]) -> None:
        """Store the non-empty secret strings to redact."""
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secrets in the message in place; never drops records."""
        if self._secrets:
            message = record.getMessage()
            for secret in self._secrets:
                message = message.replace(secret, REDACTED)
            record.msg = message
            record.args = None
        return True


def _secret_values(settings: Settings) -> list[str]:
    """Collect the plain-text values of every SecretStr field on settings."""
    values: list[str] = []
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        if isinstance(value, SecretStr):
            values.append(value.get_secret_value())
    return values


def configure_logging(settings: Settings) -> None:
    """Install a JSON stdout handler on the root logger at `settings.log_level`.

    uvicorn's loggers are routed through the same handler, so server and access logs are
    JSON and redacted as well. Idempotent: calling it again replaces the previously installed
    handler.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    secrets = _secret_values(settings)
    handler.setFormatter(JsonFormatter(secrets))
    handler.addFilter(SecretRedactingFilter(secrets))
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    _route_server_loggers_to_root()
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _route_server_loggers_to_root() -> None:
    """Drop uvicorn's own handlers so its records propagate to the redacting root handler."""
    for name in _SERVER_LOGGERS:
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True
