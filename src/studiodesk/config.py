"""Application settings loaded from environment variables and an optional `.env` file."""

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, IPvAnyNetwork, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from studiodesk import __version__

GITHUB_REPO_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$"
RATE_LIMIT_PATTERN = r"^[1-9]\d{0,5}/(second|minute|hour|day)$"
# ElevenLabs agent ids (e.g. `agent_01abc...`): kept URL-safe because they go in a query.
ELEVENLABS_AGENT_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
VOICE_LLM_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
# Public https URL without userinfo, port, query or fragment (shown on the main page).
PUBLIC_URL_PATTERN = r"^https://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._~/-]*)?$"
DEFAULT_SOURCE_REPO_URL = "https://github.com/raksharakyan/studiodesk"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
SLACK_WEBHOOK_HOST = "hooks.slack.com"
# The sample values in `.env.example`. A `.env` copied without filling them in must behave
# as "not configured" (never send a placeholder as a credential or call a fake host).
# GITHUB_REPO=your-org/your-repo is not listed: an empty repo is a validation error, and
# filing is off anyway while GITHUB_TOKEN is its placeholder.
PLACEHOLDER_VALUES = frozenset(
    {
        "gsk_your-groq-key-here",
        "sk-ant-your-key-here",
        "https://your-cluster.cloud.qdrant.io:6333",
        "your-qdrant-api-key",
        "github_pat_your-token-here",
        "your-elevenlabs-api-key",
    }
)
DEFAULT_LLM_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-sonnet-5-5",
}


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
        # Validation errors must never echo raw inputs: they may be secrets.
        hide_input_in_errors=True,
    )

    # Application
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_version: str = Field(default=__version__, min_length=1, max_length=64)
    max_request_bytes: int = Field(default=64_000, gt=0, le=10_000_000)
    # Public source repository linked from the main page and GET /api/demo-info.
    source_repo_url: str = Field(
        default=DEFAULT_SOURCE_REPO_URL, max_length=255, pattern=PUBLIC_URL_PATTERN
    )

    # LLM
    # Groq is the default (free tier); Anthropic is an optional second provider.
    llm_provider: Literal["groq", "anthropic"] = "groq"
    # Unset = the provider's default from DEFAULT_LLM_MODELS (see `resolved_llm_model`).
    llm_model: str | None = Field(default=None, min_length=1, max_length=128)
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    # Sampling temperature (used by Groq; structured Anthropic calls use the default).
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    # Groq only: reasoning effort for the (reasoning) model, e.g. openai/gpt-oss-120b.
    llm_reasoning_effort: Literal["low", "medium", "high"] = "medium"
    # Output effort for the model (`output_config.effort`).
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Reasoning models spend part of this budget on reasoning tokens.
    llm_max_tokens: int = Field(default=4096, gt=0, le=32_000)
    llm_timeout_s: float = Field(default=30.0, gt=0, le=600)
    # Overall budget per LLM call including retries; no schema or rate-limit retry starts
    # once it is spent.
    llm_request_deadline_s: float = Field(default=75.0, gt=0, le=600)
    # Process-wide cap on concurrent LLM calls; a call waits at most llm_concurrency_wait_s
    # for a slot, else 503 with Retry-After: llm_busy_retry_after_s.
    llm_max_concurrency: int = Field(default=4, ge=1, le=64)
    llm_concurrency_wait_s: float = Field(default=2.0, ge=0, le=30)
    llm_busy_retry_after_s: int = Field(default=5, ge=1, le=300)
    # Interactive routes: retry a transient failure at most this often, and only when the
    # wait (Retry-After or backoff) is <= llm_max_retry_wait_s; longer waits fail fast (503).
    llm_max_retries: int = Field(default=1, ge=0, le=3)
    llm_max_retry_wait_s: float = Field(default=10.0, gt=0, le=60)
    # Server-side retry on a substitute model when the requested model declines.
    llm_refusal_fallback: bool = True

    # Agent: answers, duplicate detection, routing
    answer_top_k: int = Field(default=6, ge=1, le=20)
    dup_search_k: int = Field(default=5, ge=1, le=20)
    # Calibrated with the pinned MiniLM model; see studiodesk.agent.duplicates.
    dup_candidate_threshold: float = Field(default=0.48, ge=0.0, le=1.0)
    dup_auto_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    routing_k: int = Field(default=7, ge=1, le=50)

    # Confirmed actions
    action_ttl_s: int = Field(default=900, ge=30, le=86_400)
    action_max_pending: int = Field(default=1000, ge=1, le=100_000)
    # Kill switch for outward actions. Default: on in dev/test, OFF in prod unless set
    # explicitly (see _actions_off_in_prod_by_default): there is no auth yet.
    actions_enabled: bool = True
    # Maximum GitHub issues filed per UTC day (per process).
    actions_max_per_day: int = Field(default=20, ge=1, le=10_000)

    # Vector store / embeddings
    qdrant_url: str | None = Field(default=None, max_length=2048)
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(
        default="studiodesk", min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_-]+$"
    )
    qdrant_timeout_s: int = Field(default=30, gt=0, le=300)
    # Points per upsert request; small batches keep each request under the client timeout.
    qdrant_upsert_batch_size: int = Field(default=32, gt=0, le=1024)
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
    ask_rate_limit: str = Field(default="10/minute", pattern=RATE_LIMIT_PATTERN)
    bugs_rate_limit: str = Field(default="10/minute", pattern=RATE_LIMIT_PATTERN)
    actions_rate_limit: str = Field(default="5/minute", pattern=RATE_LIMIT_PATTERN)
    # Reverse proxies (IPs or CIDRs) whose X-Forwarded-For entries are believed when
    # resolving the client IP for rate limiting. Empty = trust no proxy, use the peer IP.
    # From the environment: comma-separated, e.g. TRUSTED_PROXY_IPS=10.0.0.0/8,192.0.2.7
    trusted_proxy_ips: Annotated[list[IPvAnyNetwork], NoDecode] = Field(
        default_factory=list, max_length=64
    )

    # GitHub Issues
    github_token: SecretStr | None = None
    github_repo: str | None = Field(default=None, pattern=GITHUB_REPO_PATTERN)
    github_timeout_s: float = Field(default=15.0, gt=0, le=120)

    # Slack
    slack_webhook_url: SecretStr | None = None
    slack_timeout_s: float = Field(default=10.0, gt=0, le=120)

    # ElevenLabs voice agent. Voice is enabled only when both key and agent id are set.
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_agent_id: str | None = Field(default=None, pattern=ELEVENLABS_AGENT_ID_PATTERN)
    elevenlabs_timeout_s: float = Field(default=10.0, gt=0, le=120)
    # Per-client-IP limit for GET /voice/session (each call mints a signed session URL).
    voice_rate_limit: str = Field(default="10/minute", pattern=RATE_LIMIT_PATTERN)
    # Kill switch for GET /voice/session. Default: on in dev/test, OFF in prod unless set
    # explicitly (see _voice_off_in_prod_by_default): each session costs ElevenLabs credits.
    voice_enabled: bool = True
    # Maximum voice sessions minted per UTC day (per process); beyond it 503 + Retry-After.
    voice_max_sessions_per_day: int = Field(default=30, ge=1, le=100_000)
    # LLM the ElevenLabs-hosted agent runs on; used only by scripts/setup_voice_agent.py.
    voice_llm: str = Field(default="gemini-2.0-flash", pattern=VOICE_LLM_PATTERN)

    @property
    def resolved_llm_model(self) -> str:
        """`llm_model` if set, else the default model of `llm_provider`."""
        return self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def llm_api_key(self) -> SecretStr | None:
        """The API key of the selected `llm_provider` (None if not configured)."""
        return self.groq_api_key if self.llm_provider == "groq" else self.anthropic_api_key

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        """Accept log levels in any case (e.g. `info`)."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("trusted_proxy_ips", mode="before")
    @classmethod
    def _split_proxy_list(cls, value: object) -> object:
        """Accept a comma-separated string (env var) as well as a list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _prod_requires_qdrant_url(self) -> "Settings":
        """In prod, refuse to start without a remote Qdrant (no embedded local store)."""
        if self.app_env == "prod" and not self.qdrant_url:
            raise ValueError("qdrant_url is required when app_env is prod")
        return self

    @model_validator(mode="after")
    def _actions_off_in_prod_by_default(self) -> "Settings":
        """Disable outward actions in prod unless ACTIONS_ENABLED was set explicitly."""
        if self.app_env == "prod" and "actions_enabled" not in self.model_fields_set:
            object.__setattr__(self, "actions_enabled", False)
        return self

    @model_validator(mode="after")
    def _voice_off_in_prod_by_default(self) -> "Settings":
        """Disable voice sessions in prod unless VOICE_ENABLED was set explicitly."""
        if self.app_env == "prod" and "voice_enabled" not in self.model_fields_set:
            object.__setattr__(self, "voice_enabled", False)
        return self

    @model_validator(mode="after")
    def _ordered_dup_thresholds(self) -> "Settings":
        """Require `dup_candidate_threshold <= dup_auto_threshold`."""
        if self.dup_candidate_threshold > self.dup_auto_threshold:
            raise ValueError("dup_candidate_threshold must not exceed dup_auto_threshold")
        return self

    @field_validator(
        "groq_api_key",
        "anthropic_api_key",
        "qdrant_url",
        "qdrant_api_key",
        "github_token",
        "slack_webhook_url",
        "llm_model",
        "elevenlabs_api_key",
        "elevenlabs_agent_id",
        mode="before",
    )
    @classmethod
    def _empty_or_placeholder_is_unset(cls, value: object) -> object:
        """Treat an empty value (e.g. `SLACK_WEBHOOK_URL=`) or an unchanged `.env.example`
        placeholder (e.g. `your-elevenlabs-api-key`) as not configured."""
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw, str) and (not raw.strip() or raw.strip() in PLACEHOLDER_VALUES):
            return None
        return value

    @field_validator("slack_webhook_url")
    @classmethod
    def _require_slack_webhook_host(cls, value: SecretStr | None) -> SecretStr | None:
        """Accept only `https://hooks.slack.com/...` (exact host, no userinfo or port)."""
        if value is None:
            return None
        parts = urlsplit(value.get_secret_value())
        if (
            parts.scheme != "https"
            or parts.netloc != SLACK_WEBHOOK_HOST
            or parts.hostname != SLACK_WEBHOOK_HOST
            or parts.username is not None
            or parts.password is not None
            or parts.path in ("", "/")
        ):
            raise ValueError(f"slack_webhook_url must be an https://{SLACK_WEBHOOK_HOST}/ URL")
        return value

    @field_validator("qdrant_url")
    @classmethod
    def _require_https_qdrant_url(cls, value: str | None) -> str | None:
        """Require an https URL (http only for localhost) without userinfo, query or fragment."""
        if value is None:
            return None
        parts = urlsplit(value)
        if not parts.hostname:
            raise ValueError("qdrant_url must be an absolute URL with a host")
        if "@" in parts.netloc or parts.username is not None or parts.password is not None:
            raise ValueError("qdrant_url must not contain credentials; use QDRANT_API_KEY")
        if parts.query or parts.fragment or "?" in value or "#" in value:
            raise ValueError("qdrant_url must not contain a query string or fragment")
        if parts.scheme == "https":
            return value
        if parts.scheme == "http" and parts.hostname in LOCAL_HOSTS:
            return value
        raise ValueError("qdrant_url must use https:// (plain http only for localhost)")
