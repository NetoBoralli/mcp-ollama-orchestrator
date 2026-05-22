"""MCP connection management, tool discovery, and validated tool execution.

`MCPHub` owns the lifetime of every server connection via a single
`AsyncExitStack`, aggregates their tools into one flat namespace for the model,
and routes each call back to the server that owns the tool — validating the
arguments against that server's published JSON Schema before they ever leave the
process.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import jsonschema
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent

from .config import ServerConfig, StdioServerConfig, StreamableHTTPServerConfig
from .exceptions import MCPConnectionError, ToolExecutionError, ToolValidationError

logger = logging.getLogger(__name__)

# Ollama/OpenAI function names are restricted to this charset. MCP tool names are
# free-form, so we sanitize when we expose them to the model.
_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


@dataclass(slots=True)
class RegisteredTool:
    """A discovered tool plus the routing/validation context to invoke it."""

    exposed_name: str  # what the model sees (sanitized, server-namespaced)
    server: str
    original_name: str  # the real tool name on the server
    description: str
    input_schema: dict[str, Any]
    session: ClientSession


class MCPHub:
    """Aggregates one or more MCP servers behind a single tool namespace."""

    def __init__(self, servers: dict[str, ServerConfig]) -> None:
        self._servers = servers
        self._stack = AsyncExitStack()
        self._tools: dict[str, RegisteredTool] = {}

    # --- Lifecycle --------------------------------------------------------- #

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Closes every session and subprocess opened during connect(), in reverse
        # order, regardless of how the run terminated.
        await self._stack.aclose()

    async def connect(self) -> None:
        """Open every configured server and discover its tools."""
        for name, cfg in self._servers.items():
            try:
                session = await self._open_session(name, cfg)
                await session.initialize()
                await self._discover(name, session)
            except Exception as exc:  # noqa: BLE001 — annotate which server failed
                raise MCPConnectionError(f"failed to connect to MCP server '{name}': {exc}") from exc
        logger.info("Connected %d server(s); %d tool(s) available", len(self._servers), len(self._tools))

    async def _open_session(self, name: str, cfg: ServerConfig) -> ClientSession:
        """Open a transport + session, registering both with the exit stack."""
        if isinstance(cfg, StdioServerConfig):
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                # Merge onto the parent env so the child still sees PATH etc.;
                # an empty dict would otherwise strip the environment entirely.
                env={**os.environ, **cfg.env} if cfg.env else None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif isinstance(cfg, StreamableHTTPServerConfig):
            # streamablehttp_client yields a third value (a session-id getter)
            # that we don't need here.
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(cfg.url, headers=cfg.headers)
            )
        else:  # pragma: no cover — exhaustiveness guard
            raise MCPConnectionError(f"unknown transport for server '{name}'")

        session = await self._stack.enter_async_context(ClientSession(read, write))
        return session

    async def _discover(self, server: str, session: ClientSession) -> None:
        """Read the server's tool catalog and register each tool."""
        result = await session.list_tools()
        for tool in result.tools:
            exposed = self._qualify(server, tool.name)
            if exposed in self._tools:
                raise MCPConnectionError(f"tool name collision after sanitization: '{exposed}'")
            self._tools[exposed] = RegisteredTool(
                exposed_name=exposed,
                server=server,
                original_name=tool.name,
                description=tool.description or "",
                # MCP guarantees inputSchema is a JSON Schema object.
                input_schema=tool.inputSchema or {"type": "object", "properties": {}},
                session=session,
            )
            logger.debug("registered tool '%s' (from %s/%s)", exposed, server, tool.name)

    @staticmethod
    def _qualify(server: str, tool_name: str) -> str:
        """Namespace + sanitize a tool name so it's collision-free and legal."""
        return _NAME_SAFE.sub("_", f"{server}-{tool_name}")[:64]

    # --- Exposure to the model -------------------------------------------- #

    def ollama_tools(self) -> list[dict[str, Any]]:
        """Render every discovered tool as an Ollama/OpenAI function schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.exposed_name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools.values()
        ]

    # --- Execution -------------------------------------------------------- #

    async def execute(self, exposed_name: str, arguments: dict[str, Any]) -> str:
        """Validate, dispatch, and stringify a single tool call.

        Raises only the *recoverable* error types — the orchestrator catches them
        and feeds the message back to the model for self-correction. The clean
        string return value is what flows into the model's context buffer.
        """
        tool = self._tools.get(exposed_name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise ToolValidationError(
                f"unknown tool '{exposed_name}'. Available tools: {available}"
            )

        # Outgoing validation: arguments must satisfy the server's schema before
        # they hit the wire. This is where most open-weight-model mistakes die.
        try:
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
        except jsonschema.ValidationError as exc:
            raise ToolValidationError(
                f"arguments for '{exposed_name}' failed schema validation: {exc.message}"
            ) from exc

        try:
            result: CallToolResult = await tool.session.call_tool(tool.original_name, arguments)
        except Exception as exc:  # noqa: BLE001 — normalize SDK/transport errors
            raise ToolExecutionError(f"tool '{exposed_name}' raised: {exc}") from exc

        text = self._stringify(result)
        if result.isError:
            # Server-reported failure: surface the body so the model can adapt.
            raise ToolExecutionError(f"tool '{exposed_name}' returned an error: {text}")
        return text

    @staticmethod
    def _stringify(result: CallToolResult) -> str:
        """Flatten MCP content blocks into a single string for the model.

        Text passes through; non-text blocks are described compactly so the model
        knows they exist without us dumping raw base64 into the context window.
        """
        parts: list[str] = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ImageContent):
                parts.append(f"[image content: {block.mimeType}]")
            elif isinstance(block, EmbeddedResource):
                parts.append(f"[embedded resource: {getattr(block.resource, 'uri', 'unknown')}]")
            else:  # forward-compatible with future content types
                parts.append(f"[unsupported content: {type(block).__name__}]")
        return "\n".join(parts).strip()
