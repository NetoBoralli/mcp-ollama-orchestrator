"""Generic, type-safe agent loop connecting self-hosted Ollama to MCP servers."""

from __future__ import annotations

from .agent_orchestrator import Agent
from .config import Settings
from .mcp_client import MCPHub
from .ollama_client import OllamaClient

__all__ = ["Agent", "MCPHub", "OllamaClient", "Settings"]
__version__ = "0.1.0"
