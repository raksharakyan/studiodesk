"""Model for `GET /api/demo-info`: public, non-secret facts the main page explains.

It deliberately carries no keys, agent ids, tokens or URLs with credentials: only the
public source repository, the sandbox issue repository and per-feature switches and caps.
"""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

DEMO_GAME: Final = "Starfall Outpost"


class FeatureStatus(BaseModel):
    """Whether a capped feature is switched on, configured, and how much is left today."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    configured: bool
    max_per_day: int = Field(ge=0)
    remaining_today: int = Field(ge=0)


class LLMStatus(BaseModel):
    """What is left today of the process-wide LLM call budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remaining_today: int = Field(ge=0)


class DemoInfo(BaseModel):
    """Body returned by `GET /api/demo-info`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    game: Literal["Starfall Outpost"] = DEMO_GAME
    source_repo_url: str
    sandbox_repo: str | None
    sandbox_issues_url: str | None
    actions: FeatureStatus
    voice: FeatureStatus
    llm: LLMStatus
