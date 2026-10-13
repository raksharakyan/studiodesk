"""The voice agent's config-as-code (`voice/agent.json`) and the API payloads built from it.

The file holds the agent's name, first message, language, system prompt, client tool
definitions, conversation limits (`conversation.max_duration_seconds`, default 180), voice
delivery (`tts`: expressive mode off, stability), turn-taking (`turn`: how long to wait
in silence and how eagerly to take the turn), the message spoken when the duration cap is
hit, an allowlist of ElevenLabs built-in system tools (only `skip_turn` is supported), and
which client-side overrides the browser may use (`client_overrides`: only `first_message`).
The agent's LLM comes from `Settings.voice_llm`; tool ids come from the workspace at setup
time. Payload shapes follow the ElevenLabs Agents API
(`POST /v1/convai/tools`, `POST /v1/convai/agents/create`).
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
# A hostname (optionally with a port) for platform_settings.auth.allowlist.
ALLOW_HOST_PATTERN = re.compile(
    r"^(?=.{1,253}(?::\d{1,5})?$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}))*(?::\d{1,5})?$"
)
MAX_ALLOW_HOSTS = 10
# Hard cap on one conversation's length; bounds the ElevenLabs cost of a single session.
DEFAULT_MAX_DURATION_S = 180
# Expressive mode makes the voice exaggerated and lets audio tags ("[happy]") leak into
# transcripts, so it is off by default.
DEFAULT_EXPRESSIVE_MODE = False
DEFAULT_STABILITY = 0.6
# Seconds of user silence before the agent takes the turn; ElevenLabs allows 1-30.
MIN_TURN_TIMEOUT_S = 1.0
MAX_TURN_TIMEOUT_S = 30.0
DEFAULT_TURN_TIMEOUT_S = MAX_TURN_TIMEOUT_S
# The only built-in system tool StudioDesk may enable. `skip_turn` lets the agent stay
# silent on its turn; every other built-in (end_call, transfers, ...) stays off, so the
# agent cannot act outside the two client tools.
BuiltInToolName = Literal["skip_turn"]
SKIP_TURN_DESCRIPTION = (
    "Stay silent and keep waiting for the user. Use it whenever the user is silent and you "
    "have already checked in once in this conversation, or when the user asks for a moment."
)
_BUILT_IN_TOOL_DESCRIPTIONS: dict[str, str] = {"skip_turn": SKIP_TURN_DESCRIPTION}


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


class VoiceTtsConfig(BaseModel):
    """Voice delivery settings (a subset of `conversation_config.tts`).

    Only these keys are sent, so the model and voice configured on the agent stay as they are.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    expressive_mode: bool = DEFAULT_EXPRESSIVE_MODE
    stability: float = Field(default=DEFAULT_STABILITY, ge=0.0, le=1.0)


class VoiceTurnConfig(BaseModel):
    """Turn-taking settings (a subset of `conversation_config.turn`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_timeout: float = Field(
        default=DEFAULT_TURN_TIMEOUT_S, ge=MIN_TURN_TIMEOUT_S, le=MAX_TURN_TIMEOUT_S
    )
    turn_eagerness: Literal["patient", "normal", "eager"] = "patient"


class VoiceClientOverridesConfig(BaseModel):
    """Client-side overrides the browser may send when it starts a session.

    Only `first_message` can be allowed. It lets the page start later calls in the same
    tab silently (`{agent: {firstMessage: ""}}`). That is safe: a client can only change
    the greeting its own browser hears, never what other users hear. The system prompt,
    language, voice/TTS, LLM and tools stay locked, so no client can change the agent's
    behaviour. Any other key is rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    first_message: StrictBool = False


class VoiceAgentConfig(BaseModel):
    """The whole `voice/agent.json` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=100)
    first_message: str = Field(min_length=1, max_length=500)
    language: str = Field(pattern=r"^[a-z]{2}$")
    prompt: str = Field(min_length=1, max_length=20_000)
    tools: list[VoiceToolConfig] = Field(min_length=1, max_length=10)
    conversation: VoiceConversationConfig = Field(default_factory=VoiceConversationConfig)
    tts: VoiceTtsConfig = Field(default_factory=VoiceTtsConfig)
    turn: VoiceTurnConfig = Field(default_factory=VoiceTurnConfig)
    # Spoken when `conversation.max_duration_seconds` is reached. Verify live: the field
    # path follows the ElevenLabs Agents API docs and has not been checked against the API.
    max_conversation_duration_message: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    built_in_tools: list[BuiltInToolName] = Field(default_factory=list, max_length=1)
    client_overrides: VoiceClientOverridesConfig = Field(default_factory=VoiceClientOverridesConfig)

    @field_validator("built_in_tools")
    @classmethod
    def _unique_built_in_tools(cls, value: list[str]) -> list[str]:
        """Each built-in tool may be enabled once."""
        if len(value) != len(set(value)):
            raise ValueError("built-in tool names must be unique")
        return value

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


def client_overrides_payload(allowed: VoiceClientOverridesConfig) -> dict[str, Any]:
    """`platform_settings.overrides.conversation_config_override` for the agent.

    Every override is sent explicitly (not left to the ElevenLabs default) and the setup
    script re-asserts it on every run. Only `first_message` may be enabled, via
    `client_overrides.first_message` in agent.json: a client can then only change the
    greeting its own browser hears (the page uses it to start later calls in a tab
    silently). The system prompt, language and voice stay `false`, and the LLM and tools
    are not overridable at all, so the browser cannot change the agent's behaviour.
    Verify field shape live: this follows the ElevenLabs Agents API docs as of M5 and has
    not been checked against the live API.
    """
    # verify field shape live (ElevenLabs may ignore unknown keys silently).
    return {
        "agent": {
            "prompt": {"prompt": False},
            "first_message": allowed.first_message,
            "language": False,
        },
        "tts": {"voice_id": False},
    }


def tool_payload(tool: VoiceToolConfig) -> dict[str, Any]:
    """The `tool_config` object for creating or updating `tool`."""
    return tool.model_dump(mode="json")


def built_in_tools_payload(names: list[BuiltInToolName]) -> dict[str, Any]:
    """`conversation_config.agent.prompt.built_in_tools` holding only the allowed tools.

    Any built-in not listed here (end_call, transfer_to_agent, transfer_to_number, ...) is
    simply absent from the payload, so it is never enabled by StudioDesk.
    """
    return {
        name: {
            "type": "system",
            "name": name,
            "description": _BUILT_IN_TOOL_DESCRIPTIONS[name],
            "params": {"system_tool_type": name},
        }
        for name in names
    }


def agent_payload(
    config: VoiceAgentConfig, *, llm: str, tool_ids: list[str], allow_hosts: list[str]
) -> dict[str, Any]:
    """The body for creating or updating the agent.

    Auth is always enabled (sessions need a server-minted signed URL). The allowlist is
    sent only when hosts are given, so an update without `--allow-host` leaves it as is.
    Every client-side override except an allowed `first_message` is disabled on each run
    (see `client_overrides_payload`).
    `tts` and `turn` carry only the keys StudioDesk manages, so a PATCH leaves their
    siblings (model, voice, other timeouts) untouched. `built_in_tools` is always sent and
    holds only the allowlisted tools (at most `skip_turn`).
    """
    auth: dict[str, Any] = {"enable_auth": True}
    if allow_hosts:
        auth["allowlist"] = [{"hostname": host} for host in allow_hosts]
    agent: dict[str, Any] = {
        "first_message": config.first_message,
        "language": config.language,
        "prompt": {
            "prompt": config.prompt,
            "llm": llm,
            "tool_ids": tool_ids,
            "built_in_tools": built_in_tools_payload(config.built_in_tools),
        },
    }
    if config.max_conversation_duration_message is not None:
        # verify live: field path per the ElevenLabs Agents API docs, not checked live.
        agent["max_conversation_duration_message"] = config.max_conversation_duration_message
    return {
        "name": config.name,
        "conversation_config": {
            "agent": agent,
            "conversation": {
                "max_duration_seconds": config.conversation.max_duration_seconds,
            },
            "tts": config.tts.model_dump(mode="json"),
            "turn": config.turn.model_dump(mode="json"),
        },
        "platform_settings": {
            "auth": auth,
            "overrides": {
                "conversation_config_override": client_overrides_payload(config.client_overrides)
            },
        },
    }
