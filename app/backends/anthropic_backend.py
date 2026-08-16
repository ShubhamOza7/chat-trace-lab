"""Direct Anthropic SDK backend."""

from __future__ import annotations

from typing import Any

import anthropic
from opentelemetry.trace import Status, StatusCode

from ..config import SETTINGS
from ..messages import Message, ToolCall, TurnResult, Usage
from .base import finish_inference_span, start_inference_span


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Neutral messages -> Anthropic wire format.

    Consecutive tool results are merged into a single user message, which is
    what the API expects when the model made parallel tool calls.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})

        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks})

        elif msg.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content,
            }
            if msg.is_error:
                block["is_error"] = True
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) and all(
                b.get("type") == "tool_result" for b in out[-1]["content"]
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})

    return out


class AnthropicBackend:
    name = "anthropic"

    def __init__(self) -> None:
        # Credentials resolve from the environment (ANTHROPIC_API_KEY, or an
        # `ant auth login` profile). Never hardcode a key.
        self.client = anthropic.Anthropic()

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> TurnResult:
        span = start_inference_span(SETTINGS.model, system, messages, self.name)
        started = time.monotonic()
        try:
            # No `thinking` parameter: on Claude Opus 5 that means adaptive
            # thinking, which is the recommended default.
            response = self.client.messages.create(
                model=SETTINGS.model,
                max_tokens=SETTINGS.max_tokens,
                system=system,
                messages=_to_anthropic_messages(messages),
                tools=tools,
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.end()
            raise

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )

        result = TurnResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            response_id=response.id,
            stop_reason=response.stop_reason,
        )
        finish_inference_span(span, result)
        return result
