"""Liveness endpoint."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from studiodesk.api.deps import get_settings
from studiodesk.config import Settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Body returned by `GET /health`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    version: str
    env: Literal["dev", "test", "prod"]


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the service is up, with its version and environment."""
    return HealthResponse(status="ok", version=settings.app_version, env=settings.app_env)
