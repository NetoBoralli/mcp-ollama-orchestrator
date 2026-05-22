"""The agent event loop: prompt -> reason -> act -> observe -> repeat.

This is the heart of the system. It is intentionally framework-free: a single
bounded `async for`-style loop you can read top to bottom and own completely.
"""

from __future__ import annotations

import logging

from .config import Settings
from .exceptions import MaxTurnsExceeded, ToolExecutionError, ToolValidationError
from .mcp_client import MCPHub
from .ollama_client import OllamaClient
from .schemas import Message, Role

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, autonomous assistant with access to external tools.\n"
    "Use a tool only when it is necessary to answer the user; otherwise reply "
    "directly. When you call a tool, supply arguments that strictly match its "
    "JSON schema. If a tool result contains an error, read it carefully and "
    "correct your next call rather than repeating the same one. When you have "
    "enough information, give a final, self-contained answer with no further "
    "tool calls."
)


class Agent:
    """Drives the dynamic tool-execution loop with full context preservation."""

    def __init__(
        self,
        ollama: OllamaClient,
        hub: MCPHub,
        settings: Settings,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._ollama = ollama
        self._hub = hub
        self._settings = settings
        self._system_prompt = system_prompt

    async def run(self, user_prompt: str) -> str:
        """Execute one full agent run and return the model's final text answer.

        The message list IS the agent's memory: every assistant turn and every
        tool observation is appended in order, so each model call sees the entire
        causal history. This is what lets the agent self-correct across turns.
        """
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            Message(role=Role.USER, content=user_prompt),
        ]

        # Tools are discovered once at connect time; the schema list is stable for
        # the run, so we build it a single time and reuse it every turn.
        tools = self._hub.ollama_tools()

        for turn in range(1, self._settings.max_agent_turns + 1):
            self._warn_if_context_pressured(messages, turn)

            response = await self._ollama.chat(messages, tools)
            assistant = response.message
            # Preserve the assistant turn verbatim — including the tool_calls it
            # requested — so the follow-up tool results have something to attach to.
            messages.append(assistant)

            # No tool calls => the model is done. This is the loop's only exit.
            if not assistant.tool_calls:
                logger.info("Run complete in %d turn(s)", turn)
                return assistant.content

            logger.info("Turn %d: model requested %d tool call(s)", turn, len(assistant.tool_calls))
            for call in assistant.tool_calls:
                observation = await self._dispatch(call.function.name, call.function.arguments)
                messages.append(
                    Message(role=Role.TOOL, content=observation, tool_name=call.function.name)
                )

        raise MaxTurnsExceeded(
            f"no final answer after {self._settings.max_agent_turns} turns; "
            "raise ORCH_MAX_AGENT_TURNS or inspect the tool loop"
        )

    async def _dispatch(self, name: str, arguments: dict[str, object]) -> str:
        """Execute one tool call, converting recoverable failures into feedback.

        The error-interception contract: a bad tool call must never crash the run.
        Instead the raw error text becomes the tool observation, so the next model
        turn can see exactly what went wrong and fix it.
        """
        try:
            return await self._hub.execute(name, arguments)  # type: ignore[arg-type]
        except (ToolValidationError, ToolExecutionError) as exc:
            logger.warning("Tool '%s' failed; feeding error back to model: %s", name, exc)
            return f"TOOL_ERROR: {exc}"

    def _warn_if_context_pressured(self, messages: list[Message], turn: int) -> None:
        """Heuristic guard against silent context truncation.

        Ollama clips anything past `num_ctx` without telling us. We can't tokenize
        exactly without the model's tokenizer, but a ~4-chars-per-token estimate is
        enough to flag when a schema-heavy buffer is approaching the configured
        window so the operator can raise `num_ctx` before answers degrade.
        """
        approx_tokens = sum(len(m.content or "") for m in messages) // 4
        if approx_tokens > self._settings.num_ctx * 0.8:
            logger.warning(
                "Turn %d: context ~%d tokens, nearing num_ctx=%d — consider increasing it",
                turn,
                approx_tokens,
                self._settings.num_ctx,
            )
