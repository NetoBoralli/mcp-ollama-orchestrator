"""MCP connection management, tool discovery, and validated tool execution.

`MCPHub` aggregates one or more MCP servers into a single flat tool namespace for
the model and routes each call back to the owning server — validating arguments
against that server's published JSON Schema before they leave the process.

Connection lifecycle
---------------------
The MCP SDK's stdio / Streamable HTTP transports each spawn their own internal
anyio task group. Decomposing those onto one app-lifetime ``AsyncExitStack`` is a
known footgun: when a connection fails (or even on ordinary shutdown) the stack
unwinds a transport whose task group lives in a different task, raising
``RuntimeError: Attempted to exit cancel scope in a different task``.

To avoid that entirely, each server runs in its own dedicated asyncio task whose
``async with`` blocks open the transport, keep it alive, and tear it down — all
within that single task. The hub coordinates startup/shutdown with events. This
is the same pattern battle-tested by the wider MCP ecosystem.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
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


def _root_cause(exc: BaseException) -> BaseException:
    """Drill through nested ExceptionGroups to the first concrete leaf cause.

    The SDK transports wrap failures in their own task-group ExceptionGroups, so
    a raw ``eg.exceptions[0]`` is often just another group. This yields the actual
    error (e.g. the underlying ``httpx.ConnectError``) for a readable message.
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


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
        self._tools: dict[str, RegisteredTool] = {}

        # Coordination primitives between the hub and the per-server tasks.
        self._supervisor: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()  # set on shutdown -> servers unwind
        self._ready = asyncio.Event()  # set once every server is connected
        self._ready_count = 0
        self._lock = asyncio.Lock()  # guards _tools / _ready_count
        self._error: BaseException | None = None  # first startup failure, if any

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
        await self.aclose()

    async def connect(self) -> None:
        """Start every server and block until all are ready (or one fails)."""
        if not self._servers:
            raise MCPConnectionError("no MCP servers configured")

        self._supervisor = asyncio.create_task(self._supervise())
        # _ready is set either by the last server signalling ready, or by the
        # supervisor's failure path — so this wait always resolves.
        await self._ready.wait()

        if self._error is not None:
            await self.aclose()
            raise MCPConnectionError(f"failed to start MCP server(s): {self._error}") from self._error

        logger.info(
            "Connected %d server(s); %d tool(s) available", len(self._servers), len(self._tools)
        )

    async def aclose(self) -> None:
        """Signal every server task to unwind and wait for them to finish."""
        self._stop.set()
        if self._supervisor is not None:
            # The supervisor records failures on self._error, so awaiting it
            # never re-raises here; we just want a clean join.
            await self._supervisor
            self._supervisor = None

    async def _supervise(self) -> None:
        """Run all per-server tasks under one TaskGroup for the hub's lifetime.

        The group stays open while servers are parked on ``_stop``; it unwinds
        cleanly once ``aclose`` sets the event. If any server raises during
        startup, the group cancels its siblings and we record the cause.
        """
        try:
            async with asyncio.TaskGroup() as tg:
                for name, cfg in self._servers.items():
                    tg.create_task(self._serve(name, cfg), name=f"mcp:{name}")
        except* Exception as eg:
            # Keep the first concrete cause for a readable error message.
            self._error = _root_cause(eg)
        finally:
            # Unblock connect() whether startup succeeded or failed.
            self._ready.set()

    async def _serve(self, name: str, cfg: ServerConfig) -> None:
        """Own one server connection start-to-finish inside a single task.

        Entering and exiting the transport's context managers in the same task is
        precisely what anyio's cancel scopes require — which is what makes this
        teardown-safe where a shared AsyncExitStack is not.
        """
        async with self._transport(name, cfg) as streams:
            # stdio yields (read, write); Streamable HTTP yields a third element
            # (a session-id getter) we don't need.
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                await self._discover(name, session)
                await self._signal_ready()
                # Hold the connection open until the hub is shut down.
                await self._stop.wait()

    def _transport(self, name: str, cfg: ServerConfig) -> Any:
        """Return (without entering) the transport context manager for a server."""
        if isinstance(cfg, StdioServerConfig):
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                # Merge onto the parent env so the child still sees PATH etc.;
                # an empty dict would otherwise strip the environment entirely.
                env={**os.environ, **cfg.env} if cfg.env else None,
            )
            return stdio_client(params)
        if isinstance(cfg, StreamableHTTPServerConfig):
            return streamablehttp_client(cfg.url, headers=cfg.headers)
        raise MCPConnectionError(f"unknown transport for server '{name}'")  # pragma: no cover

    async def _discover(self, server: str, session: ClientSession) -> None:
        """Read the server's tool catalog and register each tool."""
        result = await session.list_tools()
        async with self._lock:
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

    async def _signal_ready(self) -> None:
        """Mark one server connected; release connect() once all are up."""
        async with self._lock:
            self._ready_count += 1
            if self._ready_count >= len(self._servers):
                self._ready.set()

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
