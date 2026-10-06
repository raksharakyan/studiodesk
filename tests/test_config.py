"""Settings loading, defaults, validation and secret masking."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from studiodesk.config import Settings

SECRET_FIELDS = (
    "anthropic_api_key",
    "qdrant_api_key",
    "github_token",
    "slack_webhook_url",
    "elevenlabs_api_key",
)
ALL_ENV_VARS = (
    "APP_ENV",
    "LOG_LEVEL",
    "APP_VERSION",
    "MAX_REQUEST_BYTES",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "EMBEDDING_MODEL",
    "GITHUB_TOKEN",
    "GITHUB_REPO",
    "SLACK_WEBHOOK_URL",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_AGENT_ID",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip relevant env vars and run from an empty dir so no real .env is read."""
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults() -> None:
    """Defaults match the documented M1 configuration."""
    s = Settings(_env_file=None)

    assert s.app_env == "dev"
    assert s.log_level == "INFO"
    assert s.llm_provider == "anthropic"
    assert s.llm_model == "claude-sonnet-5-5"
    assert s.qdrant_collection == "studiodesk"
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.max_request_bytes == 64_000


def test_all_external_keys_optional() -> None:
    """Settings build with no external integration configured."""
    s = Settings(_env_file=None)

    for name in SECRET_FIELDS:
        assert getattr(s, name) is None
    assert s.qdrant_url is None
    assert s.github_repo is None
    assert s.elevenlabs_agent_id is None


def test_loads_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values come from environment variables (case-insensitive names)."""
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("LLM_MODEL", "claude-test")
    monkeypatch.setenv("QDRANT_COLLECTION", "bugs_v2")
    monkeypatch.setenv("github_repo", "starfall/outpost")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.setenv("MAX_REQUEST_BYTES", "1024")

    s = Settings(_env_file=None)

    assert s.app_env == "prod"
    assert s.llm_model == "claude-test"
    assert s.qdrant_collection == "bugs_v2"
    assert s.github_repo == "starfall/outpost"
    assert s.max_request_bytes == 1024
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-from-env"


def test_loads_from_env_file(tmp_path: Path) -> None:
    """An explicit .env file is read."""
    env_file = tmp_path / "custom.env"
    env_file.write_text("APP_ENV=test\nLLM_MODEL=from-file\nUNRELATED_KEY=ignored\n")

    s = Settings(_env_file=env_file)

    assert s.app_env == "test"
    assert s.llm_model == "from-file"


def test_env_var_beats_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Environment variables take precedence over the .env file."""
    env_file = tmp_path / "custom.env"
    env_file.write_text("LLM_MODEL=from-file\n")
    monkeypatch.setenv("LLM_MODEL", "from-env")

    assert Settings(_env_file=env_file).llm_model == "from-env"


def test_settings_are_frozen() -> None:
    """Settings are immutable after construction."""
    s = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        s.app_env = "prod"  # type: ignore[misc]


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_secret_values_never_revealed(field: str) -> None:
    """SecretStr fields are masked in repr, str and JSON dumps."""
    secret = f"super-secret-{field}-value"
    s = Settings(_env_file=None, **{field: secret})  # type: ignore[arg-type]

    value = getattr(s, field)
    assert isinstance(value, SecretStr)
    assert value.get_secret_value() == secret
    for rendered in (repr(s), str(s), s.model_dump_json(), str(s.model_dump()), repr(value)):
        assert secret not in rendered


@pytest.mark.parametrize(
    "repo", ["owner/name", "starfall-studio/starfall.outpost", "a/b", "Owner9/repo_name-2"]
)
def test_github_repo_accepts_owner_name(repo: str) -> None:
    """Valid owner/name values are accepted."""
    assert Settings(_env_file=None, github_repo=repo).github_repo == repo


@pytest.mark.parametrize(
    "repo",
    [
        "owner",
        "owner/",
        "/name",
        "owner/name/extra",
        "-owner/name",
        "owner name/repo",
        "https://github.com/owner/name",
        "owner/na me",
        "",
        "o" * 40 + "/name",
    ],
)
def test_github_repo_rejects_bad_values(repo: str) -> None:
    """Malformed github_repo values are rejected."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, github_repo=repo)


@pytest.mark.parametrize("env", ["production", "staging", "PROD", ""])
def test_invalid_app_env_rejected(env: str) -> None:
    """app_env only accepts dev/test/prod."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env=env)  # type: ignore[arg-type]


def test_invalid_app_env_from_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad APP_ENV env var fails at construction time."""
    monkeypatch.setenv("APP_ENV", "staging")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("level", ["info", "Info", "INFO", "debug", "warning", "critical"])
def test_log_level_case_insensitive(level: str) -> None:
    """log_level accepts any case and normalises to upper case."""
    assert Settings(_env_file=None, log_level=level).log_level == level.upper()  # type: ignore[arg-type]


def test_log_level_from_env_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOG_LEVEL=debug from the environment is normalised."""
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert Settings(_env_file=None).log_level == "DEBUG"


@pytest.mark.parametrize("level", ["verbose", "TRACE", ""])
def test_invalid_log_level_rejected(level: str) -> None:
    """Unknown log levels are rejected."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level=level)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, 10_000_001])
def test_max_request_bytes_bounds(value: int) -> None:
    """max_request_bytes must be within (0, 10_000_000]."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_request_bytes=value)


@pytest.mark.parametrize("name", ["bad name", "bad/name", "", "x" * 256])
def test_qdrant_collection_validated(name: str) -> None:
    """qdrant_collection is restricted to a safe character set and length."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_collection=name)


def test_unsupported_llm_provider_rejected() -> None:
    """Only the anthropic provider is supported in M1."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="openai")  # type: ignore[arg-type]
