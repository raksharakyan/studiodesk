"""App factory and /health endpoint behaviour."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from studiodesk import __version__
from studiodesk.api.deps import get_settings
from studiodesk.config import Settings
from studiodesk.main import create_app


def test_health_returns_ok_with_injected_settings(client: TestClient) -> None:
    """GET /health returns 200 and reflects the injected Settings."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "9.9.9", "env": "test"}


@pytest.mark.parametrize("env", ["dev", "test", "prod"])
def test_health_reports_each_env(env: str) -> None:
    """The env field mirrors whichever app_env the app was built with."""
    settings = Settings(
        _env_file=None,
        app_env=env,  # type: ignore[arg-type]
        app_version="1.2.3",
        qdrant_url="https://example.cloud.qdrant.io",
    )
    response = TestClient(create_app(settings)).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.2.3", "env": env}


def test_create_app_overrides_get_settings(settings: Settings) -> None:
    """Explicit settings are installed as an override of the get_settings dependency."""
    app = create_app(settings)

    assert get_settings in app.dependency_overrides
    assert app.dependency_overrides[get_settings]() is settings
    assert app.version == "9.9.9"


def test_create_app_without_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit settings, the app uses get_settings() and adds no override."""
    get_settings.cache_clear()
    monkeypatch.chdir("/")  # avoid picking up a stray .env
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_VERSION", "7.7.7")
    try:
        app = create_app()
        assert get_settings not in app.dependency_overrides
        response = TestClient(app).get("/health")
        assert response.json() == {"status": "ok", "version": "7.7.7", "env": "test"}
    finally:
        get_settings.cache_clear()


def test_default_version_is_package_version() -> None:
    """Without an override, app_version defaults to the package __version__."""
    assert Settings(_env_file=None).app_version == __version__


def test_docs_available_in_dev() -> None:
    """Interactive docs are served outside prod."""
    app = create_app(Settings(_env_file=None, app_env="dev"))
    assert TestClient(app).get("/docs").status_code == 200


def test_docs_disabled_in_prod() -> None:
    """Interactive docs are not served in prod."""
    app = create_app(
        Settings(_env_file=None, app_env="prod", qdrant_url="https://example.cloud.qdrant.io")
    )
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


@pytest.mark.parametrize(("env", "expected"), [("dev", 200), ("test", 200), ("prod", 404)])
def test_openapi_schema_only_outside_prod(env: str, expected: int) -> None:
    """The OpenAPI schema is served in dev/test and hidden in prod."""
    app = create_app(
        Settings(
            _env_file=None,
            app_env=env,  # type: ignore[arg-type]
            qdrant_url="https://example.cloud.qdrant.io",
        )
    )
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == expected
    if expected == 200:
        assert "/health" in response.json()["paths"]


def test_unknown_route_returns_404(client: TestClient) -> None:
    """Unknown paths return a JSON 404."""
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_health_rejects_wrong_method(client: TestClient) -> None:
    """Only GET is routed for /health."""
    assert client.post("/health").status_code == 405


def test_health_response_never_contains_secrets() -> None:
    """Configured secrets never leak into the /health body."""
    settings = Settings(
        _env_file=None, app_env="test", anthropic_api_key=SecretStr("sk-ant-SUPERSECRET")
    )
    response = TestClient(create_app(settings)).get("/health")

    assert "SUPERSECRET" not in response.text
