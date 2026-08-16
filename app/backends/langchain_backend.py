"""LangChain backend (ChatAnthropic).

Same conversation, same tools, same span shape as the direct-SDK backend — the
only difference is the framework in between. That is what makes the two
comparable when you are judging what a tracing product captures.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from opentelemetry.trace import Status, StatusCode

from .. import prismtrace_setup as prismtrace
from ..config import SETTINGS
from ..messages import Message, ToolCall, TurnResult, Usage
from .base import finish_inference_span, start_inference_span


def _to_langchain_messages(system: str, messages: list[Message]) -> list[BaseMessage]:
    out: list[BaseMessage] = [SystemMessage(content=system)]
    for msg in messages:
        if msg.role == "user":
            out.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            out.append(
                AIMessage(
                    content=msg.content,
                    tool_calls=[
                        {"name": c.name, "args": c.arguments, "id": c.id, "type": "tool_call"}
                        for c in msg.tool_calls
                    ],
                )
            )
        elif msg.role == "tool":
            out.append(
                ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or "",
                    status="error" if msg.is_error else "success",
                )
            )
    return out


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


class LangChainBackend:
    name = "langchain"

    def __init__(self) -> None:
        from langchain_anthropic import ChatAnthropic

        self.llm = ChatAnthropic(
            model=SETTINGS.model,
            max_tokens=SETTINGS.max_tokens,
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> TurnResult:
        span = start_inference_span(SETTINGS.model, system, messages, self.name)

        # PRISMtrace's real LangChain integration: a callback handler on the
        # invocation. One handler per conversation — it takes session_id at
        # construction, so a shared handler would merge every conversation into
        # a single trajectory. None unless PRISMTRACE_API_KEY is set.
        config: dict[str, Any] = {}
        handler = prismtrace.langchain_handler(session_id) if session_id else None
        if handler is not None:
            config["callbacks"] = [handler]

        try:
            model = self.llm.bind_tools(tools) if tools else self.llm
            reply: AIMessage = model.invoke(
                _to_langchain_messages(system, messages), config=config or None
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.end()
            raise

        meta = reply.response_metadata or {}
        usage_meta = reply.usage_metadata or {}
        input_details = usage_meta.get("input_token_details") or {}

        usage = Usage(
            input_tokens=usage_meta.get("input_tokens", 0),
            output_tokens=usage_meta.get("output_tokens", 0),
            cache_read_tokens=input_details.get("cache_read", 0),
            cache_write_tokens=input_details.get("cache_creation", 0),
        )

        result = TurnResult(
            text=_extract_text(reply.content),
            tool_calls=[
                ToolCall(id=c.get("id") or "", name=c["name"], arguments=dict(c.get("args") or {}))
                for c in (reply.tool_calls or [])
            ],
            usage=usage,
            model=meta.get("model") or meta.get("model_name") or SETTINGS.model,
            response_id=meta.get("id") or reply.id,
            stop_reason=meta.get("stop_reason"),
        )
        finish_inference_span(span, result)
        if handler is not None:
            prismtrace.flush(session_id)
        return result
