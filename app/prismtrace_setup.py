"""Optional PRISMtrace instrumentation.

Entirely opt-in: with `PRISMTRACE_API_KEY` unset every function here is a no-op,
so the app, the test suite and CI behave exactly as they did before. This sits
alongside the OpenTelemetry tracing in `app/tracing.py` rather than replacing
it — running both is what lets you compare what each product captures.

Two paths, because the SDK offers different shapes for each backend:

* **LangChain** -> `PRISMtraceCallbackHandler`, attached per conversation via
  `config={"callbacks": [...]}`. This is the SDK's real integration.
* **Anthropic SDK** -> a direct POST to `/api/traces`. The SDK does ship a
  `ClaudeAgentTracer`, but its `run()` owns the tool-calling loop, which this
  app already owns in `app/chat.py`; and `PRISMtrace.trace_llm()` has no
  `session_id` parameter, so traces sent through it never group into a
  trajectory. The HTTP contract does take `session_id`, so it is used here.

Failures never propagate. An observability backend being down must not take the
chat down with it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .messages import Message

log = logging.getLogger(__name__)

API_KEY = os.getenv("PRISMTRACE_API_KEY", "").strip()
PROJECT_ID = os.getenv("PRISMTRACE_PROJECT_ID", "").strip()
# Note: the SDK's ClaudeAgentTracer defaults to the *production* host. This app
# always sends the host explicitly so staging traffic cannot leak to production.
HOST = os.getenv(
    "PRISMTRACE_HOST", "https://prismtrace-staging.up.railway.app"
).strip().rstrip("/")
AGENT_NAME = os.getenv("PRISMTRACE_AGENT_NAME", "chat-trace-lab").strip()

_TIMEOUT_SECONDS = 5.0


def enabled() -> bool:
    """True only when both credentials are present in the environment."""
    return bool(API_KEY and PROJECT_ID)


def status() -> dict[str, Any]:
    """Config summary for /healthz. Never includes the key itself."""
    return {
        "enabled": enabled(),
        "host": HOST if enabled() else None,
        "project_id": PROJECT_ID if enabled() else None,
        "agent_name": AGENT_NAME if enabled() else None,
    }


# --------------------------------------------------------------------------
# LangChain path
# --------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {}


def langchain_handler(session_id: str) -> Any | None:
    """One `PRISMtraceCallbackHandler` per conversation.

    The handler takes its `session_id` at construction, so a single shared
    handler would collapse every conversation into one trajectory. Hence the
    per-session cache.
    """
    if not enabled():
        return None
    if session_id in _HANDLERS:
        return _HANDLERS[session_id]

    try:
        from prismtrace import PRISMtraceCallbackHandler
    except ImportError:
        log.warning("PRISMTRACE_API_KEY is set but prismtrace-sdk is not installed.")
        return None

    try:
        handler = PRISMtraceCallbackHandler(
            api_key=API_KEY,
            project_id=PROJECT_ID,
            host=HOST,
            session_id=session_id,
            agent_name=AGENT_NAME,
        )
    except Exception as exc:
        log.warning("Could not build PRISMtrace handler: %s", exc)
        return None

    _HANDLERS[session_id] = handler
    return handler


def flush(session_id: str) -> None:
    handler = _HANDLERS.get(session_id)
    if handler is None:
        return
    try:
        handler.flush()
    except Exception as exc:
        log.warning("PRISMtrace flush failed: %s", exc)


# --------------------------------------------------------------------------
# Anthropic SDK path
# --------------------------------------------------------------------------


def _wire_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Neutral history -> the {role, content} list /api/traces expects."""
    out: list[dict[str, str]] = []
    for msg in messages:
        content = msg.content
        if msg.role == "assistant" and msg.tool_calls and not content:
            content = "[tool calls: " + ", ".join(c.name for c in msg.tool_calls) + "]"
        if content:
            out.append({"role": msg.role, "content": content})
    return out


def emit_trace(
    *,
    session_id: str,
    model: str,
    messages: list[Message],
    output: str,
    latency_ms: int,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    """POST one trace. Swallows every error by design."""
    if not enabled():
        return

    try:
        import httpx

        response = httpx.post(
            f"{HOST}/api/traces",
            headers={
                "Content-Type": "application/json",
                # Not a Bearer token — this API rejects Authorization headers.
                "X-PRISMtrace-Key": API_KEY,
            },
            json={
                "project_id": PROJECT_ID,
                "model": model,
                "input_messages": _wire_messages(messages),
                "output_message": output,
                "latency_ms": latency_ms,
                "session_id": session_id,
                "agent_name": AGENT_NAME,
                "token_count_input": tokens_in,
                "token_count_output": tokens_out,
            },
            timeout=_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            log.warning(
                "PRISMtrace ingest %s: %s", response.status_code, response.text[:200]
            )
    except Exception as exc:
        log.warning("PRISMtrace ingest failed: %s", exc)
