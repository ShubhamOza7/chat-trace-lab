"""Scripted conversations.

Running these gives the tracing product a spread of trace shapes to render
without you typing anything: plain turns, single and parallel tool calls, a
failing tool call the model has to recover from, a long multi-turn session, and
one turn that raises.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chat import ChatSession, InjectedFailure, TurnOutcome


@dataclass
class Scenario:
    name: str
    description: str
    turns: list[str]


SCENARIOS: list[Scenario] = [
    Scenario(
        name="plain_qa",
        description="No tools. The simplest possible trace: one turn, one model call.",
        turns=["In one sentence, what does EIRP stand for?"],
    ),
    Scenario(
        name="single_tool",
        description="One tool call, then a final answer.",
        turns=["What's the centre frequency of channel 36 in the 5 GHz band?"],
    ),
    Scenario(
        name="parallel_tools",
        description="Two tool calls in one turn — check the product renders them side by side.",
        turns=[
            "Give me the centre frequency of 2.4 GHz channel 11, and convert 23 dBm to milliwatts."
        ],
    ),
    Scenario(
        name="tool_error_recovery",
        description="The tool raises, the model gets an error result and recovers.",
        turns=["What's the centre frequency of channel 99 in the 2.4 GHz band?"],
    ),
    Scenario(
        name="multi_turn",
        description="Four turns in one session — tests session grouping and context growth.",
        turns=[
            "I'm setting up a 6 GHz AP. What's the centre frequency of channel 37?",
            "And channel 53?",
            "How far apart are those two in MHz?",
            "If my conducted power is 24 dBm, what's that in milliwatts?",
        ],
    ),
    Scenario(
        name="injected_error",
        description="A turn that raises before reaching the model — an errored trace.",
        turns=["/fail"],
    ),
]

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


@dataclass
class ScenarioRun:
    scenario: str
    session_id: str
    trace_ids: list[str]
    replies: list[str]
    error: str | None = None


def run_scenario(scenario: Scenario, user_id: str = "scenario-runner") -> ScenarioRun:
    session = ChatSession(user_id=user_id, scenario=scenario.name)
    run = ScenarioRun(scenario=scenario.name, session_id=session.session_id, trace_ids=[], replies=[])

    for turn in scenario.turns:
        try:
            outcome: TurnOutcome = session.send(turn)
        except InjectedFailure as exc:
            run.error = str(exc)
            break
        if outcome.trace_id:
            run.trace_ids.append(outcome.trace_id)
        run.replies.append(outcome.reply)

    return run


def run_all(user_id: str = "scenario-runner") -> list[ScenarioRun]:
    return [run_scenario(s, user_id=user_id) for s in SCENARIOS]
