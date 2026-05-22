"""Pydantic envelopes for everything that crosses the model <-> tooling boundary.

These models are the resiliency backbone described in the requirements: every
payload from the (less-reliable) open-weight model is parsed through them before
we act on it, and every payload we send to Ollama is serialized from them.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolFunctionCall(BaseModel):
    """The `function` payload inside a single tool call."""

    name: str
    # Ollama's *native* format returns already-parsed arguments (a dict). But
    # smaller open-weight models frequently emit them as a JSON *string*, or as
    # a malformed fragment. The validator below normalizes both shapes so the
    # rest of the system only ever sees a dict.
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_arguments(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                # Surfaced as a Pydantic ValidationError -> caught by the loop ->
                # fed back to the model so it can retry with valid JSON.
                raise ValueError(f"tool arguments were not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("tool arguments must decode to a JSON object")
            return parsed
        raise ValueError(f"unsupported tool arguments type: {type(value).__name__}")


class ToolCall(BaseModel):
    """A single tool call as requested by the model."""

    function: ToolFunctionCall
    # Ollama does not require an id (it is not OpenAI). Kept optional so the same
    # model round-trips cleanly against OpenAI-compatible gateways too.
    id: str | None = None


class Message(BaseModel):
    """One entry in the conversation buffer.

    `use_enum_values` keeps `role` as a plain string on serialization, which is
    what the Ollama REST API expects.
    """

    model_config = ConfigDict(use_enum_values=True)

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    # role=="tool" only. Recent Ollama versions key tool results back to the
    # originating function via this field, which materially improves multi-tool
    # turn accuracy on open-weight models.
    tool_name: str | None = None

    def for_ollama(self) -> dict[str, Any]:
        """Serialize to the exact dict shape Ollama's /api/chat accepts."""
        return self.model_dump(exclude_none=True)


class ChatResponse(BaseModel):
    """Parsed, non-streamed response from Ollama's /api/chat.

    Unknown fields (timings, context array, etc.) are ignored by default, so the
    model stays forward-compatible with new Ollama releases.
    """

    model: str
    message: Message
    done: bool = True
    # Useful for context-budget telemetry; populated by Ollama when available.
    prompt_eval_count: int | None = None
    eval_count: int | None = None
