"""Typed error hierarchy.

The split matters for the agent loop: some errors are *recoverable by the model*
(a bad tool call it can retry next turn) and some are *fatal to the run* (the
inference engine is down). The orchestrator catches the former and feeds them
back into the context window; it lets the latter propagate.
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for every error this package raises."""


# --- Fatal: abort the run -------------------------------------------------- #


class OllamaError(OrchestratorError):
    """The inference engine is unreachable or returned an unusable response."""


class MCPConnectionError(OrchestratorError):
    """An MCP server could not be started, connected to, or initialized."""


class MaxTurnsExceeded(OrchestratorError):
    """The agent hit its turn budget without producing a final answer.

    Almost always a sign the model is looping on a tool it cannot satisfy, or
    the turn budget is simply too low for the task.
    """


# --- Recoverable: feed back to the model for self-correction --------------- #


class ToolValidationError(OrchestratorError):
    """The model requested a tool that doesn't exist, or with arguments that
    fail the server's published JSON Schema. The message is written to be
    actionable when handed back to the model verbatim."""


class ToolExecutionError(OrchestratorError):
    """The MCP server accepted the call but reported a failure (``isError``)."""
