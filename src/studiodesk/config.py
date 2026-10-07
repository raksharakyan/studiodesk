"""Application settings loaded from environment variables and an optional `.env` file."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from studiodesk import __version__

GITHUB_REPO_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$"
RATE_LIMIT_PATTERN = r"^[1-9]\d{0,5}/(second|minute|hour|day)$"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class Settings(BaseSettings):
    """Typed StudioDesk configuration.

    Every value can be set via an environment variable of the same name (case-insensitive)
    or a `.env` file. External integration keys are optional in Milestone 1.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Application
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_version: str = Field(default=__version__, min_length=1, max_length=64)
    max_request_bytes: int = Field(default=64_000, gt=0, le=10_000_000)

    # LLM
    llm_provider: Literal["anthropic"] = "anthropic"
    llm_model: str = Field(default="claude-sonnet-5-5", min_length=1, max_length=128)
    anthropic_api_key: SecretStr | None = None

    # Vector store / embeddings
    qdrant_url: str | None = Field(default=None, max_length=2048)
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(
        default="studiodesk", min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_-]+$"
    )
    qdrant_timeout_s: int = Field(default=10, gt=0, le=300)
    # Used only when qdrant_url is unset: an embedded on-disk store, or ":memory:".
    qdrant_local_path: str = Field(default=".qdrant_data", min_length=1, max_length=1024)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", min_length=1, max_length=255
    )
    # Pinned Hugging Face commit of `embedding_model`, so downloads are reproducible.
    embedding_model_revision: str = Field(
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", pattern=r"^[0-9a-f]{40}$"
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = Field(default=64, gt=0, le=1024)

    # Search API
    search_rate_limit: str = Field(default="30/minute", pattern=RATE_LIMIT_PATTERN)

    # GitHub Issues
    github_token: SecretStr | None = None
    github_repo: str | None = Field(default=None, pattern=GITHUB_REPO_PATTERN)

    # Slack
    slack_webhook_url: SecretStr | None = None

    # ElevenLabs voice agent
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_agent_id: str | None = Field(default=None, max_length=128)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Accept log levels in any case (e.g. `info`)."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("qdrant_url", mode="before")
    @classmethod
    def _empty_qdrant_url_is_unset(cls, value: object) -> object:
        """Treat `QDRANT_URL=` (empty) as not configured."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("qdrant_url")
    @classmethod
    def _require_https_qdrant_url(cls, value: str | None) -> str | None:
        """Require an https URL, except for a Qdrant running on localhost."""
        if value is None:
            return None
        parts = urlsplit(value)
        if not parts.hostname:
            raise ValueError("qdrant_url must be an absolute URL with a host")
        if parts.scheme == "https":
            return value
        if parts.scheme == "http" and parts.hostname in LOCAL_HOSTS:
            return value
        raise ValueError("qdrant_url must use https:// (plain http only for localhost)")
