"""Application settings loaded from environment variables and an optional `.env` file."""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from studiodesk import __version__

GITHUB_REPO_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$"


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
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(
        default="studiodesk", min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_-]+$"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", min_length=1, max_length=255
    )

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
