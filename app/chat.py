"""The conversation loop: one turn span per user message, tool calls beneath it."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from opentelemetry.trace import SpanKind, Status, StatusCode, use_span

from . import semconv as sc
from .backends import ChatBackend, build_backend
from .config import SETTINGS, SYSTEM_PROMPT
from .messages import Message, ToolCall, Usage, assistant, tool_result, user
from .tools import TOOL_SPECS, ToolError, execute_tool
from .tracing import current_trace_id, tracer

MAX_TOOL_ITERATIONS = 6


class InjectedFailure(RuntimeError):
    """Raised on purpose by the /fail command, to produce an errored trace."""


@dataclass
class TurnOutcome:
    reply: str
    trace_id: str | None
    turn_index: int
    tools_used: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


def _run_tool(call: ToolCall) -> Message:
    """Execute one tool call inside its own span."""
    with tracer().start_as_current_span(
        f"execute_tool {call.name}", kind=SpanKind.INTERNAL
    ) as span:
        span.set_attribute(sc.TOOL_NAME, call.name)
        span.set_attribute(sc.TOOL_CALL_ID, call.id)
        if SETTINGS.capture_content:
            span.set_attribute(
                "gen_ai.tool.call.arguments", json.dumps(call.arguments, ensure_ascii=False)
            )
        try:
            output = execute_tool(call.name, call.arguments)
        except ToolError as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            # Hand the error back to the model rather than aborting the turn —
            # recovering from a bad tool call is itself worth seeing in a trace.
            return tool_result(call.id, f"Error: {exc}", is_error=True)

        if SETTINGS.capture_content:
            span.set_attribute("gen_ai.tool.call.result", output)
        span.set_status(Status(StatusCode.OK))
        return tool_result(call.id, output)


class ChatSession:
    """A multi-turn conversation. One instance per user session."""

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str = "local-tester",
        backend: ChatBackend | None = None,
        scenario: str | None = None,
    ) -> None:
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.user_id = user_id
        self.scenario = scenario
        self.backend = backend or build_backend()
        self.history: list[Message] = []
        self.turn_index = 0

    def send(self, text: str) -> TurnOutcome:
        """Run one user turn to completion, tool calls included."""
        self.turn_index += 1
        span = tracer().start_span(f"chat.turn {self.turn_index}", kind=SpanKind.SERVER)
        span.set_attribute(sc.SESSION_ID, self.session_id)
        span.set_attribute(sc.USER_ID, self.user_id)
        span.set_attribute(sc.CONVERSATION_ID, self.session_id)
        span.set_attribute(sc.APP_TURN_INDEX, self.turn_index)
        span.set_attribute(sc.APP_BACKEND, self.backend.name)
        if self.scenario:
            span.set_attribute(sc.APP_SCENARIO, self.scenario)
        if SETTINGS.capture_content:
            span.set_attribute("app.chat.user_message", text)

        with use_span(span, end_on_exit=True, record_exception=True, set_status_on_exception=True):
            trace_id = current_trace_id()

            if text.strip() == "/fail":
                raise InjectedFailure(
                    "Injected failure from the /fail command — this turn is meant to error."
                )

            self.history.append(user(text))
            totals = Usage()
            tools_used: list[str] = []
            iterations = 0
            reply = ""
            stop_reason = None

            while iterations < MAX_TOOL_ITERATIONS:
                iterations += 1
                result = self.backend.complete(
                    SYSTEM_PROMPT, self.history, TOOL_SPECS, session_id=self.session_id
                )

                totals.input_tokens += result.usage.input_tokens
                totals.output_tokens += result.usage.output_tokens
                totals.cache_read_tokens += result.usage.cache_read_tokens
                totals.cache_write_tokens += result.usage.cache_write_tokens
                stop_reason = result.stop_reason

                self.history.append(assistant(result.text, result.tool_calls))

                if not result.tool_calls:
                    reply = result.text
                    break

                for call in result.tool_calls:
                    tools_used.append(call.name)
                    self.history.append(_run_tool(call))
            else:
                reply = (
                    "Stopped after "
                    f"{MAX_TOOL_ITERATIONS} tool rounds without a final answer."
                )
                span.set_status(Status(StatusCode.ERROR, "tool iteration limit reached"))

            if stop_reason == "refusal":
                reply = reply or "The model declined to answer this request."

            span.set_attribute(sc.APP_TOOL_ITERATIONS, iterations)
            span.set_attribute(sc.USAGE_INPUT_TOKENS, totals.input_tokens)
            span.set_attribute(sc.USAGE_OUTPUT_TOKENS, totals.output_tokens)
            if tools_used:
                span.set_attribute("app.chat.tools_used", tools_used)
            if SETTINGS.capture_content:
                span.set_attribute("app.chat.assistant_message", reply)

            return TurnOutcome(
                reply=reply,
                trace_id=trace_id,
                turn_index=self.turn_index,
                tools_used=tools_used,
                usage=totals,
                stop_reason=stop_reason,
            )
