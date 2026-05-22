# mcp-ollama-orchestrator

A generic, decoupled, production-ready agent loop that connects a **self-hosted
Ollama** model to one or more **Model Context Protocol (MCP)** servers, with the
reliability scaffolding open-weight models actually need.

## Architectural choices

| Concern | Choice | Why |
| --- | --- | --- |
| Orchestration | Native `asyncio` + custom loop | Zero framework lock-in; the tool loop is one readable file you own end-to-end. |
| Inference | Ollama `/api/chat` over `httpx` | Explicit control of the request body — notably `options.num_ctx` on every call. |
| Protocol | Official `mcp` SDK | First-class stdio + Streamable HTTP transports, dynamic tool discovery. |
| Validation | Pydantic v2 **+** `jsonschema` | Pydantic types every envelope; `jsonschema` validates tool args against the *runtime* schema each server publishes (which Pydantic can't model statically). |

## Layout

```
src/mcp_orchestrator/
├── config.py              # Settings (env) + MCP server registry (JSON)
├── schemas.py             # Pydantic envelopes: Message, ToolCall, ChatResponse
├── exceptions.py          # Fatal vs. model-recoverable error hierarchy
├── ollama_client.py       # Async Ollama client; num_ctx pinning + retries
├── mcp_client.py          # MCPHub: connect, discover, validate, execute
├── agent_orchestrator.py  # The reason -> act -> observe loop
└── main.py                # CLI / REPL entrypoint
```

## How the loop works

1. **Discover** — `MCPHub` opens every configured server (stdio or HTTP) under a
   single `AsyncExitStack`, calls `list_tools()`, and renders the catalog as
   Ollama function schemas.
2. **Reason** — the model is queried with the full message buffer + tool schemas,
   with `num_ctx` pinned so heavy JSON schemas never get clipped.
3. **Parse** — the reply is parsed through Pydantic; tool-call arguments are
   coerced from dict *or* stringified JSON (open-weight models emit both).
4. **Validate & act** — arguments are checked against the server's JSON Schema,
   then the call is routed to the owning server asynchronously.
5. **Observe** — the result is flattened to a clean string and appended to the
   buffer. **Any failure becomes a `TOOL_ERROR:` observation** instead of
   crashing, so the model self-corrects on the next turn.
6. **Repeat** until the model returns text with no tool calls (bounded by
   `max_agent_turns`).

## Quickstart

```bash
# 1. Install (editable)
pip install -e .

# 2. Pull a model with strong native tool-calling
ollama pull qwen2.5:7b        # 7-8B llama models are noticeably weaker at this

# 3. Configure
cp .env.example .env          # tweak num_ctx, model, etc.
# edit mcp_servers.json for your MCP servers

# 4. Run
orchestrate "List the files under /private/tmp and summarize them."
# ...or omit the prompt for an interactive REPL
orchestrate -v
```

> **macOS path note:** the default `mcp_servers.json` points the filesystem
> server at `/tmp`, which macOS resolves to `/private/tmp`. The server enforces
> its allowed directory against the *resolved* path, so a prompt that says `/tmp`
> comes back as `Access denied - path outside allowed directories: /tmp not in
> /private/tmp`. Refer to **`/private/tmp`** in your prompts, or point the server
> at a non-symlinked directory (e.g. `~/mcp-workdir`) to avoid the confusion.
>
> **Model note:** if the agent returns a JSON tool call as plain *text* instead
> of acting on it (e.g. `{"name": "...", "parameters": {...}}` in the final
> answer), the model has dropped out of native tool-calling — a known weakness of
> smaller open-weight models. Switch to a stronger tool-caller such as
> `qwen2.5:7b`/`qwen2.5:14b` or `mistral-nemo`.

## Context window note

Ollama's default context is small (often 2–4k tokens) and it **silently
truncates** beyond it — which quietly corrupts tool schemas and history on
schema-dense turns. This project sets `num_ctx` explicitly on every request
(default 32k; raise `ORCH_NUM_CTX` to 65536 for large toolsets). VRAM usage
scales with context length, so size it to your hardware. The loop also logs a
warning when the buffer approaches 80% of the configured window.

## Use as a library

```python
import asyncio
from mcp_orchestrator import Agent, MCPHub, OllamaClient, Settings

async def main() -> None:
    settings = Settings()
    async with OllamaClient(settings) as ollama, MCPHub(settings.server_registry()) as hub:
        agent = Agent(ollama, hub, settings)
        print(await agent.run("Your task here."))

asyncio.run(main())
```
