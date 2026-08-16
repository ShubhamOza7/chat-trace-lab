"""Backend-neutral conversation model.

The conversation is held in this shape and converted at the edge by each
backend. That is what lets the same conversation, the same tools and the same
span structure run through either the Anthropic SDK or LangChain — which is the
point of the harness: differences you see in the tracing product come from the
integration, not from the app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Set on role="tool" only:
    tool_call_id: str | None = None
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class TurnResult:
    """One model call's worth of output, in neutral form."""

    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    response_id: str | None
    stop_reason: str | None


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str, tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls or [])


def tool_result(tool_call_id: str, content: str, *, is_error: bool = False) -> Message:
    return Message(
        role="tool", content=content, tool_call_id=tool_call_id, is_error=is_error
    )


def to_json_parts(messages: list[Message]) -> str:
    """Serialise messages for the `gen_ai.*.messages` span attributes."""
    out = []
    for m in messages:
        parts: list[dict[str, Any]] = []
        if m.content:
            kind = "tool_call_response" if m.role == "tool" else "text"
            part: dict[str, Any] = {"type": kind, "content": m.content}
            if m.role == "tool":
                part["id"] = m.tool_call_id
            parts.append(part)
        for call in m.tool_calls:
            parts.append(
                {
                    "type": "tool_call",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
        out.append({"role": m.role, "parts": parts})
    return json.dumps(out, ensure_ascii=False)
