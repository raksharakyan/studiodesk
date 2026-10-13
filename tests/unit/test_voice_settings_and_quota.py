"""Voice settings (kill switch, daily cap, `.env.example` placeholders) and the daily
VoiceSessionQuota with an injected clock. No network; `.env` is never read."""

import re
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from studiodesk.config import PLACEHOLDER_VALUES, Settings
from studiodesk.voice import VoiceDailyLimitError, VoiceSessionQuota

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Variables the `.env.example` tests depend on; removed from the process env so a
# developer's shell cannot change the outcome.
EXAMPLE_ENV_VARS = (
    "APP_ENV",
    "VOICE_ENABLED",
    "VOICE_MAX_SESSIONS_PER_DAY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_AGENT_ID",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_REPO",
    "SLACK_WEBHOOK_URL",
    "LLM_PROVIDER",
    "LLM_MODEL",
)


# Prod refuses to start without a remote Qdrant; any https URL satisfies it offline.
PROD_QDRANT = "https://example.cloud.qdrant.io"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in EXAMPLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# --- voice_enabled kill switch -------------------------------------------------------


@pytest.mark.parametrize("env", ["dev", "test"])
def test_voice_enabled_by_default_outside_prod(clean_env: pytest.MonkeyPatch, env: str) -> None:
    assert Settings(_env_file=None, app_env=env).voice_enabled is True


def test_voice_disabled_by_default_in_prod(clean_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None, app_env="prod", qdrant_url=PROD_QDRANT).voice_enabled is False


def test_voice_enabled_in_prod_when_set_by_kwarg(clean_env: pytest.MonkeyPatch) -> None:
    assert (
        Settings(
            _env_file=None, app_env="prod", qdrant_url=PROD_QDRANT, voice_enabled=True
        ).voice_enabled
        is True
    )


@pytest.mark.parametrize("value", ["true", "1", "TRUE"])
def test_voice_enabled_in_prod_when_set_by_env(clean_env: pytest.MonkeyPatch, value: str) -> None:
    clean_env.setenv("APP_ENV", "prod")
    clean_env.setenv("QDRANT_URL", PROD_QDRANT)
    clean_env.setenv("VOICE_ENABLED", value)
    assert Settings(_env_file=None).voice_enabled is True


@pytest.mark.parametrize("env", ["dev", "test", "prod"])
def test_explicit_false_stays_off(clean_env: pytest.MonkeyPatch, env: str) -> None:
    clean_env.setenv("QDRANT_URL", PROD_QDRANT)
    assert Settings(_env_file=None, app_env=env, voice_enabled=False).voice_enabled is False
    clean_env.setenv("VOICE_ENABLED", "false")
    assert Settings(_env_file=None, app_env=env).voice_enabled is False


def test_voice_enabled_in_prod_via_env_file(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / "prod.env"
    env_file.write_text(f"APP_ENV=prod\nQDRANT_URL={PROD_QDRANT}\nVOICE_ENABLED=true\n")
    assert Settings(_env_file=env_file).voice_enabled is True
    env_file.write_text(f"APP_ENV=prod\nQDRANT_URL={PROD_QDRANT}\n")
    assert Settings(_env_file=env_file).voice_enabled is False


# --- voice_max_sessions_per_day ------------------------------------------------------


def test_max_sessions_default(clean_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None).voice_max_sessions_per_day == 30


@pytest.mark.parametrize("value", [1, 100_000])
def test_max_sessions_bounds_accepted(value: int) -> None:
    assert Settings(
        _env_file=None, voice_max_sessions_per_day=value
    ).voice_max_sessions_per_day == (value)


@pytest.mark.parametrize("value", [0, -1, 100_001])
def test_max_sessions_out_of_range_rejected(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, voice_max_sessions_per_day=value)


# --- .env.example placeholders -------------------------------------------------------


def test_env_example_placeholders_load_as_unset(clean_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=ENV_EXAMPLE)
    assert settings.elevenlabs_api_key is None
    assert settings.elevenlabs_agent_id is None
    assert settings.groq_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.qdrant_url is None
    assert settings.qdrant_api_key is None
    assert settings.github_token is None
    assert settings.voice_max_sessions_per_day == 30


PLACEHOLDER_FIELDS = [
    ("elevenlabs_api_key", "your-elevenlabs-api-key"),
    ("groq_api_key", "gsk_your-groq-key-here"),
    ("anthropic_api_key", "sk-ant-your-key-here"),
    ("qdrant_url", "https://your-cluster.cloud.qdrant.io:6333"),
    ("qdrant_api_key", "your-qdrant-api-key"),
    ("github_token", "github_pat_your-token-here"),
]


@pytest.mark.parametrize(("field", "placeholder"), PLACEHOLDER_FIELDS)
@pytest.mark.parametrize("as_secret", [False, True])
def test_placeholder_is_unset_as_str_or_secret(
    clean_env: pytest.MonkeyPatch, field: str, placeholder: str, as_secret: bool
) -> None:
    value: object = SecretStr(placeholder) if as_secret else placeholder
    settings = Settings(_env_file=None, **{field: value})  # type: ignore[arg-type]
    assert getattr(settings, field) is None
    padded: object = SecretStr(f"  {placeholder} ") if as_secret else f"  {placeholder} "
    assert getattr(Settings(_env_file=None, **{field: padded}), field) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "real"),
    [
        ("elevenlabs_api_key", "sk_0123456789abcdef0123456789abcdef"),
        ("groq_api_key", "gsk_realLookingKey0123456789"),
        ("anthropic_api_key", "sk-ant-api03-realLookingKey"),
        ("qdrant_api_key", "qdrant-real-key-123"),
        ("github_token", "github_pat_11AAAAAAA0realtoken"),
    ],
)
@pytest.mark.parametrize("as_secret", [False, True])
def test_non_placeholder_secret_kept(
    clean_env: pytest.MonkeyPatch, field: str, real: str, as_secret: bool
) -> None:
    value: object = SecretStr(real) if as_secret else real
    kept = getattr(Settings(_env_file=None, **{field: value}), field)  # type: ignore[arg-type]
    assert isinstance(kept, SecretStr)
    assert kept.get_secret_value() == real


def test_non_placeholder_qdrant_url_kept(clean_env: pytest.MonkeyPatch) -> None:
    url = "https://abc-123.eu-central.aws.cloud.qdrant.io:6333"
    assert Settings(_env_file=None, qdrant_url=url).qdrant_url == url


def test_every_env_example_placeholder_is_recognised() -> None:
    """Guard: a new `your-...` sample value must also be added to PLACEHOLDER_VALUES."""
    found: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=(.*)$", line)
        if match and "your-" in match.group(2):
            found[match.group(1)] = match.group(2).strip()
    found.pop("GITHUB_REPO", None)  # documented exception: empty repo is a validation error
    assert found, "expected placeholder values in .env.example"
    missing = {k: v for k, v in found.items() if v not in PLACEHOLDER_VALUES}
    assert not missing, f"placeholders not in config.PLACEHOLDER_VALUES: {missing}"


# --- VoiceSessionQuota ---------------------------------------------------------------


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


NOON = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def test_reserve_up_to_cap_then_limit() -> None:
    quota = VoiceSessionQuota(3, clock=FakeClock(NOON))
    days = [quota.reserve() for _ in range(3)]
    assert days == [NOON.date()] * 3
    assert quota.used_today() == 3
    with pytest.raises(VoiceDailyLimitError) as info:
        quota.reserve()
    assert info.value.retry_after_s == 12 * 3600
    assert quota.used_today() == 3  # a refused reserve takes nothing


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 10, 11, 23, 59, 0, 500_000, tzinfo=UTC), 60),
        (datetime(2026, 10, 11, 23, 59, 59, 999_999, tzinfo=UTC), 1),
        (datetime(2026, 10, 11, 23, 59, 59, tzinfo=UTC), 1),
        (datetime(2026, 10, 11, 0, 0, 0, tzinfo=UTC), 86_400),
    ],
)
def test_retry_after_is_seconds_until_utc_midnight(now: datetime, expected: int) -> None:
    quota = VoiceSessionQuota(1, clock=FakeClock(now))
    quota.reserve()
    with pytest.raises(VoiceDailyLimitError) as info:
        quota.reserve()
    assert info.value.retry_after_s == expected


def test_resets_next_utc_day() -> None:
    clock = FakeClock(datetime(2026, 10, 11, 23, 59, 59, tzinfo=UTC))
    quota = VoiceSessionQuota(2, clock=clock)
    quota.reserve()
    quota.reserve()
    with pytest.raises(VoiceDailyLimitError):
        quota.reserve()
    clock.now = datetime(2026, 10, 12, 0, 0, 0, tzinfo=UTC)
    assert quota.used_today() == 0
    assert quota.reserve() == clock.now.date()
    assert quota.used_today() == 1


def test_release_frees_a_slot_same_day() -> None:
    quota = VoiceSessionQuota(1, clock=FakeClock(NOON))
    day = quota.reserve()
    quota.release(day)
    assert quota.used_today() == 0
    assert quota.reserve() == day


def test_release_of_previous_day_leaves_new_count_alone() -> None:
    clock = FakeClock(datetime(2026, 10, 11, 23, 59, 59, tzinfo=UTC))
    quota = VoiceSessionQuota(5, clock=clock)
    old_day = quota.reserve()
    clock.now += timedelta(seconds=2)  # 00:00:01 next day
    quota.reserve()
    quota.reserve()
    quota.release(old_day)
    assert quota.used_today() == 2


def test_release_without_reserve_never_goes_negative() -> None:
    quota = VoiceSessionQuota(2, clock=FakeClock(NOON))
    quota.release(NOON.date())
    assert quota.used_today() == 0
    quota.reserve()
    quota.reserve()
    with pytest.raises(VoiceDailyLimitError):
        quota.reserve()


def test_concurrent_reserve_never_exceeds_cap() -> None:
    cap, workers, attempts = 25, 16, 10
    quota = VoiceSessionQuota(cap, clock=FakeClock(NOON))
    barrier = threading.Barrier(workers)
    granted: list[int] = []
    refused: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        for _ in range(attempts):
            try:
                quota.reserve()
            except VoiceDailyLimitError:
                with lock:
                    refused.append(1)
            else:
                with lock:
                    granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(granted) == cap
    assert len(refused) == workers * attempts - cap
    assert quota.used_today() == cap


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_cap_rejected(value: int) -> None:
    with pytest.raises(ValueError):
        VoiceSessionQuota(value, clock=FakeClock(NOON))


def test_limit_error_message_is_generic() -> None:
    assert str(VoiceDailyLimitError(5)) == "daily voice session limit reached"
