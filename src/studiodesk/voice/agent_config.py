"""The voice agent's config-as-code (`voice/agent.json`) and the API payloads built from it.

The file holds the agent's name, first message, language, system prompt, client tool
definitions and conversation limits (`conversation.max_duration_seconds`, default 180).
The agent's LLM comes from `Settings.voice_llm`; tool ids come from the workspace at setup
time. Payload shapes follow the ElevenLabs Agents API
(`POST /v1/convai/tools`, `POST /v1/convai/agents/create`).
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
# A hostname (optionally with a port) for platform_settings.auth.allowlist.
ALLOW_HOST_PATTERN = re.compile(
    r"^(?=.{1,253}(?::\d{1,5})?$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}))*(?::\d{1,5})?$"
)
MAX_ALLOW_HOSTS = 10
# Hard cap on one conversation's length; bounds the ElevenLabs cost of a single session.
DEFAULT_MAX_DURATION_S = 180


class VoiceToolConfig(BaseModel):
    """One client tool: runs in the browser, which calls the StudioDesk API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["client"]
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str = Field(min_length=1, max_length=1000)
    expects_response: bool
    response_timeout_secs: int = Field(ge=1, le=120)
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def _object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Require a JSON schema of `type: object` whose required keys are properties."""
        properties = value.get("properties")
        required = value.get("required", [])
        if value.get("type") != "object" or not isinstance(properties, dict):
            raise ValueError("parameters must be an object schema with properties")
        if not isinstance(required, list) or not set(required) <= set(properties):
            raise ValueError("required must list existing properties")
        return value


class VoiceConversationConfig(BaseModel):
    """Conversation limits (`conversation_config.conversation` in the agent payload)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_duration_seconds: int = Field(default=DEFAULT_MAX_DURATION_S, ge=10, le=3600)


class VoiceAgentConfig(BaseModel):
    """The whole `voice/agent.json` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=100)
    first_message: str = Field(min_length=1, max_length=500)
    language: str = Field(pattern=r"^[a-z]{2}$")
    prompt: str = Field(min_length=1, max_length=20_000)
    tools: list[VoiceToolConfig] = Field(min_length=1, max_length=10)
    conversation: VoiceConversationConfig = Field(default_factory=VoiceConversationConfig)

    @field_validator("tools")
    @classmethod
    def _unique_tool_names(cls, value: list[VoiceToolConfig]) -> list[VoiceToolConfig]:
        """Tool names identify tools in the workspace, so they must be unique."""
        names = [tool.name for tool in value]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        return value


def load_agent_config(path: Path) -> VoiceAgentConfig:
    """Read and validate the agent config file.

    Raises:
        OSError: if the file cannot be read.
        ValueError: if it is not valid JSON (pydantic `ValidationError` for bad content).
    """
    return VoiceAgentConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_allow_hosts(hosts: list[str]) -> list[str]:
    """Return the hosts lower-cased and de-duplicated, or raise ValueError on a bad one."""
    cleaned: list[str] = []
    for host in hosts:
        if not ALLOW_HOST_PATTERN.fullmatch(host):
            raise ValueError(f"invalid --allow-host value: {host!r}")
        if host.lower() not in cleaned:
            cleaned.append(host.lower())
    if len(cleaned) > MAX_ALLOW_HOSTS:
        raise ValueError(f"at most {MAX_ALLOW_HOSTS} --allow-host values")
    return cleaned


def tool_payload(tool: VoiceToolConfig) -> dict[str, Any]:
    """The `tool_config` object for creating or updating `tool`."""
    return tool.model_dump(mode="json")


def agent_payload(
    config: VoiceAgentConfig, *, llm: str, tool_ids: list[str], allow_hosts: list[str]
) -> dict[str, Any]:
    """The body for creating or updating the agent.

    Auth is always enabled (sessions need a server-minted signed URL). The allowlist is
    sent only when hosts are given, so an update without `--allow-host` leaves it as is.
    """
    auth: dict[str, Any] = {"enable_auth": True}
    if allow_hosts:
        auth["allowlist"] = [{"hostname": host} for host in allow_hosts]
    return {
        "name": config.name,
        "conversation_config": {
            "agent": {
                "first_message": config.first_message,
                "language": config.language,
                "prompt": {"prompt": config.prompt, "llm": llm, "tool_ids": tool_ids},
            },
            "conversation": {
                "max_duration_seconds": config.conversation.max_duration_seconds,
            },
        },
        "platform_settings": {"auth": auth},
    }
