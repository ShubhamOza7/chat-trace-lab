"""Shared backend contract and span-recording helpers.

Both backends emit an identically-shaped `chat <model>` span, so a tracing
product renders the Anthropic-SDK path and the LangChain path the same way.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from .. import semconv as sc
from ..config import SETTINGS
from ..messages import Message, TurnResult, to_json_parts
from ..tracing import tracer


class ChatBackend(Protocol):
    """What the conversation loop needs from a model integration."""

    name: str

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> TurnResult: ...


def start_inference_span(model: str, system: str, messages: list[Message], backend: str) -> Span:
    """Open the GenAI client span and record everything known before the call."""
    span = tracer().start_span(f"chat {model}", kind=SpanKind.CLIENT)
    span.set_attribute(sc.OPERATION_NAME, "chat")
    span.set_attribute(sc.SYSTEM, "anthropic")
    span.set_attribute(sc.PROVIDER_NAME, "anthropic")
    span.set_attribute(sc.APP_BACKEND, backend)
    span.set_attribute(sc.REQUEST_MODEL, model)
    span.set_attribute(sc.REQUEST_MAX_TOKENS, SETTINGS.max_tokens)

    if not SETTINGS.capture_content:
        return span

    # Newer convention: content as a JSON attribute.
    span.set_attribute(sc.INPUT_MESSAGES, to_json_parts(messages))
    # Older convention: content as span events. Products read one or the other.
    span.add_event(sc.EVENT_SYSTEM_MESSAGE, {"content": system})
    for msg in messages:
        if msg.role == "user":
            span.add_event(sc.EVENT_USER_MESSAGE, {"content": msg.content})
        elif msg.role == "assistant":
            attrs: dict[str, Any] = {"content": msg.content}
            if msg.tool_calls:
                attrs["tool_calls"] = json.dumps(
                    [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in msg.tool_calls
                    ],
                    ensure_ascii=False,
                )
            span.add_event(sc.EVENT_ASSISTANT_MESSAGE, attrs)
        elif msg.role == "tool":
            span.add_event(
                sc.EVENT_TOOL_MESSAGE,
                {
                    "content": msg.content,
                    "id": msg.tool_call_id or "",
                    "is_error": msg.is_error,
                },
            )
    return span


def finish_inference_span(span: Span, result: TurnResult) -> None:
    """Record the response side, then close the span."""
    span.set_attribute(sc.RESPONSE_MODEL, result.model)
    if result.response_id:
        span.set_attribute(sc.RESPONSE_ID, result.response_id)
    if result.stop_reason:
        span.set_attribute(sc.RESPONSE_FINISH_REASONS, [result.stop_reason])

    usage = result.usage
    span.set_attribute(sc.USAGE_INPUT_TOKENS, usage.input_tokens)
    span.set_attribute(sc.USAGE_OUTPUT_TOKENS, usage.output_tokens)
    if usage.cache_read_tokens:
        span.set_attribute(sc.USAGE_CACHE_READ_TOKENS, usage.cache_read_tokens)
    if usage.cache_write_tokens:
        span.set_attribute(sc.USAGE_CACHE_WRITE_TOKENS, usage.cache_write_tokens)

    if SETTINGS.capture_content:
        parts: list[dict[str, Any]] = []
        if result.text:
            parts.append({"type": "text", "content": result.text})
        for call in result.tool_calls:
            parts.append(
                {
                    "type": "tool_call",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
        span.set_attribute(
            sc.OUTPUT_MESSAGES,
            json.dumps(
                [{"role": "assistant", "parts": parts, "finish_reason": result.stop_reason}],
                ensure_ascii=False,
            ),
        )
        span.add_event(
            sc.EVENT_CHOICE,
            {
                "index": 0,
                "finish_reason": result.stop_reason or "",
                "message": json.dumps({"content": result.text}, ensure_ascii=False),
            },
        )

    # Claude Opus 5 can decline a request: HTTP 200, stop_reason "refusal".
    # Mark it on the span so refusals are visible in the tracing product rather
    # than looking like an ordinary short answer.
    if result.stop_reason == "refusal":
        span.set_status(Status(StatusCode.ERROR, "model refused the request"))
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()
