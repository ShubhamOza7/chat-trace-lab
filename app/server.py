"""HTTP front end.

    uvicorn app.server:api --reload --port 8000

Every /chat response carries the trace id of the turn, so you can paste it
straight into whichever tracing product you are evaluating.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat import ChatSession, InjectedFailure
from .config import SETTINGS
from .scenarios import SCENARIOS, SCENARIOS_BY_NAME, run_all, run_scenario
from .tracing import init_tracing, shutdown_tracing

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# In-memory only. Restarting the server drops history — fine for a test harness.
_SESSIONS: dict[str, ChatSession] = {}


@asynccontextmanager
async def _lifespan(_: FastAPI):
    init_tracing()
    yield
    shutdown_tracing()


api = FastAPI(title="chat-trace-lab", version="0.1.0", lifespan=_lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    user_id: str = "web-tester"
    backend: str | None = Field(default=None, description="anthropic | langchain")


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    trace_id: str | None
    turn_index: int
    tools_used: list[str]
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


def _get_session(req: ChatRequest) -> ChatSession:
    if req.session_id and req.session_id in _SESSIONS:
        return _SESSIONS[req.session_id]
    from .backends import build_backend

    session = ChatSession(
        session_id=req.session_id,
        user_id=req.user_id,
        backend=build_backend(req.backend),
    )
    _SESSIONS[session.session_id] = session
    return session


@api.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "backend": SETTINGS.backend,
        "model": SETTINGS.model,
        "exporter": SETTINGS.exporter,
        "otlp_endpoint": SETTINGS.otlp_endpoint or None,
        "capture_content": SETTINGS.capture_content,
    }


@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session = _get_session(req)
    try:
        outcome = session.send(req.message)
    except InjectedFailure as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    return ChatResponse(
        reply=outcome.reply,
        session_id=session.session_id,
        trace_id=outcome.trace_id,
        turn_index=outcome.turn_index,
        tools_used=outcome.tools_used,
        input_tokens=outcome.usage.input_tokens,
        output_tokens=outcome.usage.output_tokens,
        stop_reason=outcome.stop_reason,
    )


@api.get("/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    return [
        {"name": s.name, "description": s.description, "turns": len(s.turns)} for s in SCENARIOS
    ]


@api.post("/scenarios/run")
def run_scenarios(name: str | None = None) -> list[dict[str, Any]]:
    """Generate traces from the scripted conversations. Omit `name` to run all."""
    if name:
        scenario = SCENARIOS_BY_NAME.get(name)
        if scenario is None:
            raise HTTPException(status_code=404, detail=f"No scenario named {name!r}")
        runs = [run_scenario(scenario)]
    else:
        runs = run_all()
    return [run.__dict__ for run in runs]


if STATIC_DIR.is_dir():
    api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
