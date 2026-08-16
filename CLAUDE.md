# CLAUDE.md

Notes for Claude Code (or any agent) working in this repository.

## What this repo is

`chat-trace-lab` is a **test harness, not a product.** It is a small chatbot whose
purpose is to be observed by a tracing product: it holds multi-turn sessions, calls
tools, and fails on demand, emitting a trace for each turn.

Keep that in mind when changing it. Features that make it a better *chatbot* are
usually not the point; features that make it produce more varied, more realistic
**traces** are.

## Conventions

- Python 3.10+. No formatter is enforced; match the surrounding style.
- `scripts/selftest.py` must pass with no API key and no network. Anything that
  breaks that breaks CI.
- Secrets live in `.env`, which is **gitignored**. `.env.example` carries key
  **names** only. Never commit a real key.
- Two model backends (`anthropic`, `langchain`) are kept deliberately symmetric.
  If you add behaviour to one, add it to the other, or the comparison the harness
  exists for stops being valid.

## Layout

See the table in `README.md`. The parts that matter most:

- `app/messages.py` — backend-neutral conversation model. Both backends convert
  to their own wire format at the edge.
- `app/chat.py` — owns the tool-calling loop and opens the per-turn span.
- `app/tracing.py` — the only place OpenTelemetry is configured.
- `app/prismtrace_setup.py` — the only place PRISMtrace is configured.

## Tracing

Two independent tracing systems run side by side. That is intentional — comparing
what each captures is the reason this repo exists. Neither is required for the app
to work.

### 1. OpenTelemetry (vendor-neutral)

Configured in `app/tracing.py`, driven by `TRACE_EXPORTER`,
`OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME`. Emits GenAI
semantic-convention spans over OTLP/HTTP. See `README.md` for per-vendor setup.

### 2. PRISMtrace

Configured in `app/prismtrace_setup.py`. Environment variables:

| Variable | Purpose |
| --- | --- |
| `PRISMTRACE_API_KEY` | `pt-sk-…`. **Absent = every PRISMtrace path no-ops.** |
| `PRISMTRACE_PROJECT_ID` | Target project UUID |
| `PRISMTRACE_HOST` | Defaults to the staging host |
| `PRISMTRACE_AGENT_NAME` | Label on traces; defaults to `chat-trace-lab` |

Where it is wired:

- **`app/chat.py`** — per user turn, posts one trace to `/api/traces` with a
  client-minted `trace_id`, then its span tree to `/api/spans/ingest` under the
  same id: one `llm` span per model call, one `tool` span per tool execution.
  This lives in `chat.py`, not in a backend, so both backends emit the same
  shape.
- **LangChain backend** (`app/backends/langchain_backend.py`) — additionally
  attaches a `PRISMtraceCallbackHandler` per invocation via
  `config={"callbacks": [...]}`, then flushes. This is PRISMtrace's own
  framework integration; on this backend you will therefore see both what the
  handler captures and what the explicit instrumentation captures. That
  comparison is deliberate.

Tool spans have to be posted explicitly because this app runs its own tool loop
in `chat.py` — tools are never invoked *through* LangChain, so the callback
handler never sees them. Without `/api/spans/ingest`, tool calls would be
invisible to PRISMtrace on both backends.

Three things that are easy to get wrong:

1. **Auth is `X-PRISMtrace-Key`, not `Authorization: Bearer`.** A Bearer header is
   a dashboard session and 401s with an API key.
2. **`session_id` is what turns individual traces into a trajectory.** It is wired
   to `ChatSession.session_id`. The handler takes it at *construction*, so handlers
   are cached per session — one shared handler would merge every conversation into
   a single trajectory.
3. **`ClaudeAgentTracer` defaults its `endpoint` to the *production* host**, not
   staging. This repo never relies on that default; the host is always passed
   explicitly. It is also not used for the Anthropic backend at all, because its
   `run()` wants to own the tool-calling loop that `app/chat.py` already owns, and
   `PRISMtrace.trace_llm()` has no `session_id` parameter.

PRISMtrace failures are logged and swallowed. An observability backend being down
must never take the chat down with it.
