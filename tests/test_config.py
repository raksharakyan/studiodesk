"""Settings loading, defaults, validation and secret masking."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from studiodesk.config import Settings

SECRET_FIELDS = (
    "groq_api_key",
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
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_TIMEOUT_S",
    "QDRANT_UPSERT_BATCH_SIZE",
    "QDRANT_LOCAL_PATH",
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "SEARCH_RATE_LIMIT",
    "TRUSTED_PROXY_IPS",
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
    assert s.llm_provider == "groq"
    assert s.llm_model is None
    assert s.resolved_llm_model == "llama-3.3-70b-versatile"
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
    monkeypatch.setenv("QDRANT_URL", "https://example.cloud.qdrant.io")
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


# Values are valid for each field's validators (Slack URLs must be hooks.slack.com).
SECRET_SAMPLES = {
    "groq_api_key": "super-secret-groq_api_key-value",
    "anthropic_api_key": "super-secret-anthropic_api_key-value",
    "qdrant_api_key": "super-secret-qdrant_api_key-value",
    "github_token": "super-secret-github_token-value",
    "slack_webhook_url": "https://hooks.slack.com/services/T000/B000/fakesupersecretvalue",
    "elevenlabs_api_key": "super-secret-elevenlabs_api_key-value",
}


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_secret_values_never_revealed(field: str) -> None:
    """SecretStr fields are masked in repr, str and JSON dumps."""
    secret = SECRET_SAMPLES[field]
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


# --- qdrant_url (M2) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://abc-123.eu-central.aws.cloud.qdrant.io",
        "https://abc.cloud.qdrant.io:6333",
        "http://localhost:6333",
        "http://127.0.0.1:6333",
        "http://[::1]:6333",
    ],
)
def test_qdrant_url_accepts_https_or_local_http(url: str) -> None:
    assert Settings(_env_file=None, qdrant_url=url).qdrant_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://abc.cloud.qdrant.io",
        "http://localhost.evil.io:6333",
        "ftp://abc.cloud.qdrant.io",
        "abc.cloud.qdrant.io",
        "https://",
        "localhost:6333",
        "https://" + "a" * 2050,
    ],
)
def test_qdrant_url_rejects_insecure_or_malformed(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_url=url)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_qdrant_url_means_unset(value: str) -> None:
    assert Settings(_env_file=None, qdrant_url=value).qdrant_url is None


def test_qdrant_url_error_does_not_echo_credentials() -> None:
    """The validation message itself must not repeat a URL that may embed credentials."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, qdrant_url="http://user:pa55word@remote.example")

    errors = excinfo.value.errors(include_input=False, include_url=False)
    assert "pa55word" not in str(errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_model_revision", "main"),
        ("embedding_model_revision", "1110a243"),
        ("embedding_batch_size", 0),
        ("embedding_batch_size", 1025),
        ("qdrant_timeout_s", 0),
        ("embedding_device", "tpu"),
    ],
)
def test_m2_settings_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})  # type: ignore[arg-type]


def test_m2_defaults() -> None:
    s = Settings(_env_file=None)

    assert s.search_rate_limit == "30/minute"
    assert s.embedding_batch_size == 64
    assert s.embedding_device == "cpu"
    assert len(s.embedding_model_revision) == 40


# --- slack_webhook_url host and prod-requires-qdrant_url validators ----------------------


def test_slack_webhook_on_hooks_slack_com_accepted() -> None:
    """Valid https://hooks.slack.com webhook URLs pass the host validator."""
    url = "https://hooks.slack.com/services/T000/B000/fakewebhooktoken"
    s = Settings(_env_file=None, slack_webhook_url=url)  # type: ignore[arg-type]

    assert s.slack_webhook_url is not None
    assert s.slack_webhook_url.get_secret_value() == url


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.slack.com/services/T000/B000/fake",
        "https://hooks.slack.example/services/T000/B000/fake",
        "https://evil.io/services/T000/B000/fake",
        "https://hooks.slack.com.evil.io/services/T000/B000/fake",
        "https://evil.io/hooks.slack.com/services/T000/B000/fake",
        "https://hooks.slack.com@evil.io/services/T000/B000/fake",
        "https://evilhooks.slack.com/services/T000/B000/fake",
        "hooks.slack.com/services/T000/B000/fake",
        "not a url",
    ],
)
def test_slack_webhook_rejects_other_hosts_http_and_lookalikes(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, slack_webhook_url=url)  # type: ignore[arg-type]


def test_prod_without_qdrant_url_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="prod")


def test_prod_with_qdrant_url_accepted() -> None:
    s = Settings(_env_file=None, app_env="prod", qdrant_url="https://abc.cloud.qdrant.io")

    assert s.app_env == "prod"


# --- qdrant_url: no userinfo, query or fragment ------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pw@abc.cloud.qdrant.io",
        "https://user@abc.cloud.qdrant.io",
        "https://@abc.cloud.qdrant.io",
        "http://user:pw@localhost:6333",
        "https://abc.cloud.qdrant.io?q=1",
        "https://abc.cloud.qdrant.io/?api-key=zz",
        "https://abc.cloud.qdrant.io?",
        "https://abc.cloud.qdrant.io#frag",
        "https://abc.cloud.qdrant.io/path#",
        "https://abc.cloud.qdrant.io:6333/#",
    ],
)
def test_qdrant_url_rejects_userinfo_query_fragment(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://abc.cloud.qdrant.io:6333",
        "https://abc.cloud.qdrant.io/prefix/path",
        "https://abc.cloud.qdrant.io:443/qdrant/",
        "http://localhost:6333/base",
    ],
)
def test_qdrant_url_accepts_port_and_path(url: str) -> None:
    assert Settings(_env_file=None, qdrant_url=url).qdrant_url == url


@pytest.mark.parametrize(
    ("url", "fragments"),
    [
        ("https://alice:hunter2@xyzhost.example", ["alice", "hunter2", "xyzhost"]),
        ("https://xyzhost.example/p?apikey=topsecret", ["xyzhost", "apikey", "topsecret"]),
        ("https://xyzhost.example#tok3n", ["xyzhost", "tok3n"]),
        ("http://xyzhost.example:6333/seg", ["xyzhost", "6333", "/seg"]),
    ],
)
def test_qdrant_url_error_contains_no_part_of_url(url: str, fragments: list[str]) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, qdrant_url=url)

    # str/repr are what tracebacks and logs show; errors(include_input=False) is what the CLI
    # prints. (`.json()` / `.errors()` with defaults still carry `input` by pydantic design.)
    safe_errors = excinfo.value.errors(include_input=False, include_url=False, include_context=True)
    rendered = " ".join([str(excinfo.value), repr(excinfo.value), str(safe_errors)])
    for fragment in fragments:
        assert fragment not in rendered


def test_slack_error_contains_no_part_of_url() -> None:
    url = "https://evil-hostname.example/services/T9/B9/tok3nvalue"
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, slack_webhook_url=url)  # type: ignore[arg-type]

    safe_errors = excinfo.value.errors(include_input=False, include_url=False)
    rendered = f"{excinfo.value} {excinfo.value!r} {safe_errors}"
    assert "tok3nvalue" not in rendered
    assert "evil-hostname" not in rendered
