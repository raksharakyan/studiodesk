"""Create or update the StudioDesk ElevenLabs voice agent from `voice/agent.json` (idempotent).

Syncs the client tools by name (update if present, else create), then creates the agent
when ELEVENLABS_AGENT_ID is unset or updates that agent otherwise, with authentication
enabled (sessions need a signed URL from GET /voice/session). The agent's LLM is
VOICE_LLM. Requests go only to api.elevenlabs.io. The API key is never printed.

    uv run python scripts/setup_voice_agent.py --dry-run     # print payloads, no network
    uv run python scripts/setup_voice_agent.py [--allow-host localhost ...]

Exit codes: 0 ok, 1 ElevenLabs API error, 2 configuration error.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from studiodesk.config import Settings
from studiodesk.voice.agent_config import (
    VoiceAgentConfig,
    agent_payload,
    load_agent_config,
    tool_payload,
    validate_allow_hosts,
)
from studiodesk.voice.client import ElevenLabsClient, ElevenLabsError, build_http_client
from studiodesk.voice.provision import SyncResult, sync_agent

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "voice" / "agent.json"
EXIT_API_ERROR = 1
EXIT_CONFIG_ERROR = 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="agent JSON file")
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="HOST",
        help="hostname allowed to start conversations (repeatable); omitted = unchanged",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the payloads without calling ElevenLabs"
    )
    return parser.parse_args(argv)


def _config_error(message: str) -> int:
    """Print a configuration error and return its exit code."""
    print(f"error: {message}", file=sys.stderr)
    return EXIT_CONFIG_ERROR


def _validation_errors(exc: ValidationError, prefix: str) -> str:
    """Field paths and messages of a ValidationError, never the input values."""
    return "; ".join(
        f"{prefix}{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )


def load_settings() -> Settings | str:
    """Settings from the environment / `.env`, or an error message."""
    try:
        return Settings()
    except ValidationError as exc:
        return f"invalid settings: {_validation_errors(exc, '')}"


def load_config(path: Path) -> VoiceAgentConfig | str:
    """The validated agent config, or an error message."""
    try:
        return load_agent_config(path)
    except ValidationError as exc:
        return f"invalid {path.name}: {_validation_errors(exc, '')}"
    except (OSError, ValueError) as exc:
        return f"cannot read {path}: {type(exc).__name__}"


def print_dry_run(config: VoiceAgentConfig, settings: Settings, allow_hosts: list[str]) -> None:
    """Print what would be sent, with placeholder tool ids. No network calls."""
    for tool in config.tools:
        print(f"# tool {tool.name}: POST /v1/convai/tools or PATCH /v1/convai/tools/{{id}}")
        print(json.dumps({"tool_config": tool_payload(tool)}, indent=2))
    placeholders = [f"<id of {tool.name}>" for tool in config.tools]
    payload: dict[str, Any] = agent_payload(
        config, llm=settings.voice_llm, tool_ids=placeholders, allow_hosts=allow_hosts
    )
    target = (
        f"PATCH /v1/convai/agents/{settings.elevenlabs_agent_id}"
        if settings.elevenlabs_agent_id
        else "POST /v1/convai/agents/create"
    )
    print(f"# agent: {target}")
    print(json.dumps(payload, indent=2))


def print_result(result: SyncResult) -> None:
    """Print the synced tools, the agent id and the `.env` line."""
    for tool in result.tools:
        print(f"Tool {tool.name}: {tool.action} ({tool.id})")
    print(f"Agent {result.agent_action}: {result.agent_id}")
    print("Add this line to .env:")
    print(f"ELEVENLABS_AGENT_ID={result.agent_id}")


def run(args: argparse.Namespace, settings: Settings, http: httpx.Client | None = None) -> int:
    """Validate inputs, then dry-run or sync. Returns the exit code.

    Args:
        args: Parsed command line.
        settings: Loaded settings (key, agent id, LLM, timeout).
        http: Outbound client to use (tests pass a MockTransport one); built if None.
    """
    config = load_config(args.config)
    if isinstance(config, str):
        return _config_error(config)
    try:
        allow_hosts = validate_allow_hosts(args.allow_host)
    except ValueError as exc:
        return _config_error(str(exc))
    if args.dry_run:
        print_dry_run(config, settings, allow_hosts)
        return 0
    if settings.elevenlabs_api_key is None:
        return _config_error("set ELEVENLABS_API_KEY (see .env.example)")

    owned = http is None
    client_http = build_http_client(settings.elevenlabs_timeout_s) if http is None else http
    try:
        client = ElevenLabsClient(
            client_http, settings.elevenlabs_api_key, settings.elevenlabs_timeout_s
        )
        result = sync_agent(
            client,
            config,
            agent_id=settings.elevenlabs_agent_id,
            llm=settings.voice_llm,
            allow_hosts=allow_hosts,
        )
    except ElevenLabsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.detail:
            print(f"elevenlabs: {exc.detail}", file=sys.stderr)
        return EXIT_API_ERROR
    finally:
        if owned:
            client_http.close()
    print_result(result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: load settings and run. Returns the exit code."""
    args = parse_args(argv)
    settings = load_settings()
    if isinstance(settings, str):
        return _config_error(settings)
    return run(args, settings)


if __name__ == "__main__":
    sys.exit(main())
