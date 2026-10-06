"""Smoke test: the app starts and /health responds."""

from fastapi.testclient import TestClient

from studiodesk.config import Settings
from studiodesk.main import create_app


def test_health_returns_ok_with_injected_settings() -> None:
    """GET /health returns 200 and reflects the injected Settings."""
    settings = Settings(_env_file=None, app_env="test", app_version="9.9.9")
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "9.9.9", "env": "test"}
