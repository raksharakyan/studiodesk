"""Minimal structured (JSON-lines) logging configured from Settings.

Secret values from Settings are redacted from every log record as a defence in depth;
code should still never pass secrets to a logger.
"""

import json
import logging
import sys
from datetime import UTC, datetime

from pydantic import SecretStr

from studiodesk.config import Settings

REDACTED = "**********"
_HANDLER_NAME = "studiodesk-json"
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys() | {"message", "asctime"}
)


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record, including any `extra=` fields, to JSON."""
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
        return json.dumps(payload, default=str)


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

    Idempotent: calling it again replaces the previously installed handler.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SecretRedactingFilter(_secret_values(settings)))
    root.addHandler(handler)
    root.setLevel(settings.log_level)
