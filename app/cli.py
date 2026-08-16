"""Terminal chat REPL.

    python -m app.cli
    python -m app.cli --backend langchain
"""

from __future__ import annotations

import argparse
import sys

from .chat import ChatSession, InjectedFailure
from .config import SETTINGS
from .tracing import init_tracing, shutdown_tracing

HELP = """
Commands:
  /new       start a fresh session (new session id, empty history)
  /session   print the current session id and turn count
  /fail      raise on purpose, to produce an errored trace
  /help      this text
  /quit      exit
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the traced test bot.")
    parser.add_argument("--backend", choices=["anthropic", "langchain"], default=None)
    parser.add_argument("--user", default="local-tester", help="user id recorded on each trace")
    args = parser.parse_args(argv)

    init_tracing()
    from .backends import build_backend

    backend = build_backend(args.backend)
    session = ChatSession(user_id=args.user, backend=backend)

    print(f"chat-trace-lab | backend={backend.name} model={SETTINGS.model}")
    print(f"session={session.session_id}  (/help for commands)\n")

    try:
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not text:
                continue
            if text in {"/quit", "/exit"}:
                break
            if text == "/help":
                print(HELP)
                continue
            if text == "/new":
                session = ChatSession(user_id=args.user, backend=backend)
                print(f"new session={session.session_id}\n")
                continue
            if text == "/session":
                print(f"session={session.session_id} turns={session.turn_index}\n")
                continue

            try:
                outcome = session.send(text)
            except InjectedFailure as exc:
                print(f"bot> [errored on purpose] {exc}\n")
                continue
            except Exception as exc:  # surfaced, and already recorded on the span
                print(f"bot> [error] {type(exc).__name__}: {exc}\n")
                continue

            print(f"bot> {outcome.reply}")
            detail = f"      trace={outcome.trace_id} tokens_in={outcome.usage.input_tokens} tokens_out={outcome.usage.output_tokens}"
            if outcome.tools_used:
                detail += f" tools={','.join(outcome.tools_used)}"
            print(detail + "\n")
    finally:
        shutdown_tracing()

    return 0


if __name__ == "__main__":
    sys.exit(main())
