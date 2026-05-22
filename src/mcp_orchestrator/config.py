"""Configuration: Ollama endpoint, model + context parameters, and MCP servers.

Core runtime knobs come from the environment (12-factor friendly), while the set
of MCP servers comes from a JSON file modeled on the familiar Claude Desktop
`mcpServers` format, so existing server definitions drop in unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# MCP server definitions (discriminated union on `transport`).
# --------------------------------------------------------------------------- #


class StdioServerConfig(BaseModel):
    """A server we launch as a subprocess and speak to over stdio."""

    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    # Extra env vars (e.g. API keys) merged onto the parent environment at spawn.
    env: dict[str, str] = Field(default_factory=dict)


class StreamableHTTPServerConfig(BaseModel):
    """A remote server reachable over MCP's Streamable HTTP transport."""

    transport: Literal["streamable_http"] = "streamable_http"
    url: str
    # e.g. {"Authorization": "Bearer ${TOKEN}"} — useful for hosted MCP servers.
    headers: dict[str, str] = Field(default_factory=dict)


ServerConfig = Annotated[
    StdioServerConfig | StreamableHTTPServerConfig,
    Field(discriminator="transport"),
]

# A reusable parser for the {name: server} mapping in the JSON file.
_ServersAdapter: TypeAdapter[dict[str, ServerConfig]] = TypeAdapter(dict[str, ServerConfig])


def load_servers(path: Path) -> dict[str, ServerConfig]:
    """Parse and validate the MCP server registry file.

    Raises a clear error early (at startup) rather than letting a malformed
    server definition surface deep inside connection logic.
    """
    if not path.exists():
        raise FileNotFoundError(f"MCP server config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Tolerate the Claude Desktop top-level {"mcpServers": {...}} wrapper.
    if isinstance(raw, dict) and "mcpServers" in raw:
        raw = raw["mcpServers"]
    return _ServersAdapter.validate_python(raw)


# --------------------------------------------------------------------------- #
# Runtime settings.
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Process-wide settings, populated from env (prefix ``ORCH_``) and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="ORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Inference engine ---
    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"

    # --- Context window management ---
    # THE critical knob. Ollama's default context is small (often 2k-4k) and it
    # *silently truncates* anything beyond it — which quietly destroys tool
    # schemas and conversation history on heavy-JSON turns. We set it explicitly
    # on every request. Scale to 32768 / 65536 for schema-dense toolsets, bearing
    # in mind VRAM grows with context length.
    num_ctx: int = 32768

    # Deterministic tool selection: keep temperature at/near 0 for the agent loop.
    temperature: float = 0.0

    # --- Reliability ---
    request_timeout_s: float = 120.0
    ollama_max_retries: int = 3
    ollama_backoff_base_s: float = 0.5

    # Hard ceiling on tool-use iterations per run; guards against infinite loops
    # where a model repeatedly calls a tool it can't satisfy.
    max_agent_turns: int = 10

    # --- MCP ---
    mcp_config_path: Path = Path("mcp_servers.json")

    def server_registry(self) -> dict[str, ServerConfig]:
        return load_servers(self.mcp_config_path)
