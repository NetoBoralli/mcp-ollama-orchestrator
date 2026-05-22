"""Thin async client over Ollama's /api/chat endpoint.

Deliberately minimal: one responsibility — turn a typed message buffer + tool
schemas into a typed `ChatResponse`, with explicit context configuration and
bounded retries on transient transport/server failures.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import ValidationError

from .config import Settings
from .exceptions import OllamaError
from .schemas import ChatResponse, Message

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async wrapper around a self-hosted Ollama instance.

    Use as an async context manager so the underlying connection pool is shared
    across the whole agent run and cleanly torn down afterwards.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_host,
            timeout=settings.request_timeout_s,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Run one non-streamed chat completion.

        `tools` is the Ollama/OpenAI-style function schema list (built by the MCP
        hub from live tool discovery). We always pin `num_ctx` here so a large
        schema payload can never get clipped by Ollama's small default context.
        """
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [m.for_ollama() for m in messages],
            "stream": False,
            "options": {
                "num_ctx": self._settings.num_ctx,
                "temperature": self._settings.temperature,
            },
        }
        if tools:
            payload["tools"] = tools

        data = await self._post_with_retry("/api/chat", payload)

        try:
            response = ChatResponse.model_validate(data)
        except ValidationError as exc:
            # The HTTP call succeeded but the body wasn't the shape we expect —
            # treat as fatal; retrying won't change a structurally wrong reply.
            raise OllamaError(f"Ollama returned an unparseable chat response: {exc}") from exc

        if response.prompt_eval_count is not None:
            # Cheap, high-signal telemetry for spotting context pressure.
            logger.debug(
                "ollama tokens: prompt=%s completion=%s (num_ctx=%s)",
                response.prompt_eval_count,
                response.eval_count,
                self._settings.num_ctx,
            )
        return response

    async def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exponential backoff on transient failures.

        Retries connection errors and 5xx responses (engine warming up, model
        loading, transient overload). Does not retry 4xx — those are our bug.
        """
        last_exc: Exception | None = None
        for attempt in range(self._settings.ollama_max_retries):
            try:
                resp = await self._client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code < 500:
                    raise OllamaError(
                        f"Ollama rejected the request ({exc.response.status_code}): "
                        f"{exc.response.text}"
                    ) from exc
            except httpx.TransportError as exc:
                last_exc = exc

            backoff = self._settings.ollama_backoff_base_s * (2**attempt)
            logger.warning(
                "Ollama call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                self._settings.ollama_max_retries,
                last_exc,
                backoff,
            )
            await asyncio.sleep(backoff)

        raise OllamaError(
            f"Ollama unreachable after {self._settings.ollama_max_retries} attempts at "
            f"{self._settings.ollama_host}"
        ) from last_exc
