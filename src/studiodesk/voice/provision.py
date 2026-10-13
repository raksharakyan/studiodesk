"""Idempotently create or update the voice agent's client tools and the agent itself.

Tools are matched by name against the workspace's tool list: an existing tool is updated
in place, a missing one is created. The agent is created when no agent id is configured,
otherwise updated. Running it twice converges to the same state.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from studiodesk.voice.agent_config import VoiceAgentConfig, agent_payload, tool_payload
from studiodesk.voice.client import ElevenLabsClient, ElevenLabsError, RemoteTool

SyncAction = Literal["created", "updated"]


class SyncedTool(BaseModel):
    """A tool after syncing: its name, workspace id and what was done."""

    model_config = ConfigDict(frozen=True)

    name: str
    id: str
    action: SyncAction


class SyncResult(BaseModel):
    """Outcome of a full sync."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    agent_action: SyncAction
    tools: list[SyncedTool]


def _existing_by_name(tools: list[RemoteTool], wanted: set[str]) -> dict[str, str]:
    """Map each wanted tool name to its id; refuse ambiguous (duplicate) names."""
    found: dict[str, str] = {}
    for tool in tools:
        if tool.name not in wanted:
            continue
        if tool.name in found:
            raise ElevenLabsError(
                f"several workspace tools are named {tool.name}; delete the extra ones"
            )
        found[tool.name] = tool.id
    return found


def sync_tools(client: ElevenLabsClient, config: VoiceAgentConfig) -> list[SyncedTool]:
    """Create or update every tool in `config`, in file order."""
    existing = _existing_by_name(client.list_tools(), {tool.name for tool in config.tools})
    synced: list[SyncedTool] = []
    for tool in config.tools:
        payload = tool_payload(tool)
        tool_id = existing.get(tool.name)
        if tool_id is None:
            synced.append(
                SyncedTool(name=tool.name, id=client.create_tool(payload), action="created")
            )
        else:
            client.update_tool(tool_id, payload)
            synced.append(SyncedTool(name=tool.name, id=tool_id, action="updated"))
    return synced


def sync_agent(
    client: ElevenLabsClient,
    config: VoiceAgentConfig,
    *,
    agent_id: str | None,
    llm: str,
    allow_hosts: list[str],
) -> SyncResult:
    """Sync the tools, then create the agent (no `agent_id`) or update it."""
    tools = sync_tools(client, config)
    payload = agent_payload(
        config, llm=llm, tool_ids=[tool.id for tool in tools], allow_hosts=allow_hosts
    )
    if agent_id is None:
        return SyncResult(
            agent_id=client.create_agent(payload), agent_action="created", tools=tools
        )
    client.update_agent(agent_id, payload)
    return SyncResult(agent_id=agent_id, agent_action="updated", tools=tools)
