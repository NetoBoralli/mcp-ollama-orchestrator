"""CLI entrypoint wiring the three layers together.

    Settings -> {OllamaClient, MCPHub} -> Agent -> run()

Run a one-shot prompt:
    orchestrate "What files are in /tmp?"

Or start an interactive REPL:
    orchestrate
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .agent_orchestrator import Agent
from .config import Settings
from .exceptions import OrchestratorError
from .mcp_client import MCPHub
from .ollama_client import OllamaClient


async def _run(prompt: str | None) -> None:
    settings = Settings()
    servers = settings.server_registry()

    # Both clients are async context managers; entering them here guarantees the
    # Ollama pool and every MCP subprocess are torn down on any exit path.
    async with OllamaClient(settings) as ollama, MCPHub(servers) as hub:
        agent = Agent(ollama, hub, settings)

        if prompt is not None:
            print(await agent.run(prompt))
            return

        print("MCP/Ollama orchestrator ready. Ctrl-D or 'exit' to quit.\n")
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print()
                return
            if user_input.lower() in {"exit", "quit"}:
                return
            if not user_input:
                continue
            try:
                print(f"\nagent> {await agent.run(user_input)}\n")
            except OrchestratorError as exc:
                # Keep the REPL alive on a fatal run error; the engine may recover.
                print(f"\n[run failed] {exc}\n", file=sys.stderr)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Self-hosted MCP + Ollama agent loop.")
    parser.add_argument("prompt", nargs="?", help="One-shot prompt; omit for interactive REPL.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        asyncio.run(_run(args.prompt))
    except OrchestratorError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    cli()
