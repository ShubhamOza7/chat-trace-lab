#!/usr/bin/env python3
"""Offline self-test — no API key, no network, no model calls.

Checks the parts that break silently: tool execution, the two message
converters, and that a span actually reaches an exporter. Run it after cloning,
before you spend a token.

    python scripts/selftest.py
"""

from __future__ import annotations

import json
import os
import sys

# Force a self-contained config before app modules read the environment.
os.environ.setdefault("TRACE_EXPORTER", "none")
os.environ.setdefault("CAPTURE_MESSAGE_CONTENT", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from app.backends.anthropic_backend import _to_anthropic_messages  # noqa: E402
from app.backends.langchain_backend import _extract_text, _to_langchain_messages  # noqa: E402
from app.messages import ToolCall, assistant, to_json_parts, tool_result, user  # noqa: E402
from app.tools import TOOL_SPECS, ToolError, execute_tool  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        failures.append(label)


CONVERSATION = [
    user("Centre frequency of channel 36 in 5 GHz, and 23 dBm in mW?"),
    assistant(
        "Let me look both up.",
        [
            ToolCall(id="toolu_1", name="wifi_channel_info", arguments={"channel": 36, "band": "5"}),
            ToolCall(id="toolu_2", name="dbm_to_milliwatts", arguments={"dbm": 23}),
        ],
    ),
    tool_result("toolu_1", "Channel 36 ... 5180 MHz ..."),
    tool_result("toolu_2", "23 dBm = 199.526 mW."),
]


print("tools")
check("wifi_channel_info 2.4 GHz ch 6", "2437 MHz" in execute_tool("wifi_channel_info", {"channel": 6, "band": "2.4"}))
check("wifi_channel_info 5 GHz ch 36", "5180 MHz" in execute_tool("wifi_channel_info", {"channel": 36, "band": "5"}))
check("wifi_channel_info 6 GHz ch 37", "6135 MHz" in execute_tool("wifi_channel_info", {"channel": 37, "band": "6"}))
check("dbm_to_milliwatts 0 dBm", "= 1 mW" in execute_tool("dbm_to_milliwatts", {"dbm": 0}))
check("dbm_to_milliwatts -30 dBm", "0.001 mW" in execute_tool("dbm_to_milliwatts", {"dbm": -30}))

try:
    execute_tool("wifi_channel_info", {"channel": 99, "band": "2.4"})
    check("invalid channel raises ToolError", False)
except ToolError:
    check("invalid channel raises ToolError", True)

try:
    execute_tool("no_such_tool", {})
    check("unknown tool raises ToolError", False)
except ToolError:
    check("unknown tool raises ToolError", True)

check("every tool spec has an input_schema", all("input_schema" in t for t in TOOL_SPECS))


print("\nanthropic message conversion")
converted = _to_anthropic_messages(CONVERSATION)
check("three wire messages (tool results merged)", len(converted) == 3, f"got {len(converted)}")
check("assistant carries both tool_use blocks",
      sum(1 for b in converted[1]["content"] if b["type"] == "tool_use") == 2)
check("merged tool results land in one user turn",
      converted[2]["role"] == "user" and len(converted[2]["content"]) == 2)
check("tool_result blocks reference their call ids",
      {b["tool_use_id"] for b in converted[2]["content"]} == {"toolu_1", "toolu_2"})

err = _to_anthropic_messages([tool_result("toolu_9", "Error: bad channel", is_error=True)])
check("error tool result sets is_error", err[0]["content"][0].get("is_error") is True)


print("\nlangchain message conversion")
lc = _to_langchain_messages("system prompt here", CONVERSATION)
check("system prompt prepended", lc[0].type == "system")
check("one message per neutral message, plus system", len(lc) == len(CONVERSATION) + 1)
check("assistant tool calls survive", len(lc[2].tool_calls) == 2)
check("tool message keeps its call id", lc[3].tool_call_id == "toolu_1")
check("error status maps through",
      _to_langchain_messages("s", [tool_result("t", "Error: x", is_error=True)])[1].status == "error")
check("text extracted from block list",
      _extract_text([{"type": "text", "text": "hi "}, {"type": "tool_use"}, {"type": "text", "text": "there"}])
      == "hi there")
check("text extracted from plain string", _extract_text("plain") == "plain")


print("\ncontent serialisation")
parsed = json.loads(to_json_parts(CONVERSATION))
check("json round-trips", len(parsed) == 4)
check("tool_call part carries arguments",
      any(p["type"] == "tool_call" and p["arguments"] == {"channel": 36, "band": "5"}
          for p in parsed[1]["parts"]))


print("\nspan export")
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

with trace.get_tracer("selftest").start_as_current_span("chat.turn 1") as parent:
    parent.set_attribute("session.id", "sess_selftest")
    with trace.get_tracer("selftest").start_as_current_span("execute_tool wifi_channel_info"):
        pass

spans = exporter.get_finished_spans()
check("two spans exported", len(spans) == 2, f"got {len(spans)}")
check("child nests under the turn span",
      spans[0].parent is not None and spans[0].parent.span_id == spans[1].context.span_id)
check("session id recorded", spans[1].attributes.get("session.id") == "sess_selftest")


print("\nprismtrace payloads (stubbed transport, no network)")
os.environ["PRISMTRACE_API_KEY"] = "pt-sk-selftest"
os.environ["PRISMTRACE_PROJECT_ID"] = "00000000-0000-0000-0000-000000000000"

import importlib  # noqa: E402
from unittest.mock import patch  # noqa: E402

from app import prismtrace_setup  # noqa: E402

importlib.reload(prismtrace_setup)  # pick up the env vars set above

from app.chat import ChatSession  # noqa: E402
from app.messages import TurnResult, Usage  # noqa: E402


class _StubBackend:
    name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, messages, tools, session_id=None):
        self.calls += 1
        if self.calls == 1:
            return TurnResult(
                "Looking that up.",
                [
                    ToolCall("t1", "wifi_channel_info", {"channel": 36, "band": "5"}),
                    ToolCall("t2", "wifi_channel_info", {"channel": 99, "band": "2.4"}),
                ],
                Usage(10, 5),
                "claude-opus-5",
                "msg_1",
                "tool_use",
            )
        return TurnResult("Done.", [], Usage(20, 8), "claude-opus-5", "msg_2", "end_turn")


posted: dict = {}
with patch.object(prismtrace_setup, "_post", lambda path, payload: posted.update({path: payload})):
    _session = ChatSession(backend=_StubBackend())
    _outcome = _session.send("channel 36 and 99?")

check("both endpoints posted", set(posted) == {"/api/traces", "/api/spans/ingest"}, str(list(posted)))
if posted:
    _trace, _spans = posted["/api/traces"], posted["/api/spans/ingest"]
    check("spans reference the same trace_id", _trace["trace_id"] == _spans["trace_id"])
    check("session id groups the trajectory", _trace["session_id"] == _session.session_id)
    check("span per model call and per tool", len(_spans["spans"]) == 4, str(len(_spans["spans"])))
    check("tool spans present", [s["name"] for s in _spans["spans"]].count("wifi_channel_info") == 2)
    check("failed tool span marked error",
          any(s["span_type"] == "tool" and s["status"] == "error" and s["error_message"]
              for s in _spans["spans"]))
    check("every span carries the required schema fields",
          all({"name", "span_type", "start_time"} <= set(s) for s in _spans["spans"]))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) — {', '.join(failures)}")
    sys.exit(1)
print("All offline checks passed. Add ANTHROPIC_API_KEY and run `python -m app.cli` for a live turn.")
