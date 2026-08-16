# chat-trace-lab

A small chatbot built for one purpose: **to be watched by a chat-tracing product.**

It talks to Claude, calls tools, holds multi-turn sessions, and fails on demand — and it
emits an OpenTelemetry trace for every turn using the GenAI semantic conventions. Point it
at whatever tracing/observability product you are evaluating, chat at it, and see what the
product actually captures.

Two model paths ship side by side, selected by one environment variable:

| `CHAT_BACKEND` | Path |
| --- | --- |
| `anthropic` | Direct [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) |
| `langchain` | LangChain `ChatAnthropic` |

Same conversation, same tools, same span shape either way — so any difference you see in
the tracing product comes from the integration, not from the app.

---

## Quickstart

```bash
git clone <your-fork-url> chat-trace-lab && cd chat-trace-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/selftest.py        # no API key needed — verifies the plumbing
```

Then add credentials and run it:

```bash
cp .env.example .env              # set ANTHROPIC_API_KEY, pick a TRACE_EXPORTER
python -m app.cli                 # terminal chat
```

Or the web UI:

```bash
uvicorn app.server:api --reload --port 8000
```

Open <http://localhost:8000>. Every reply shows its **trace id** — paste that into your
tracing product to find the turn.

---

## Connecting a tracing product

Tracing is plain OTLP/HTTP, so most products need env vars only — no code change.
Set `TRACE_EXPORTER=otlp` (or `both` to also print spans to your terminal).

| Product | Configuration |
| --- | --- |
| **Any OTLP collector** | `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` |
| **Langfuse** | `OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel` plus `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — the Basic auth header is built for you |
| **Arize Phoenix** (local) | `docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix`, then `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006` |
| **Traceloop / OpenLLMetry** | `OTEL_EXPORTER_OTLP_ENDPOINT=https://api.traceloop.com` and `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer%20<key>` |
| **Datadog / Honeycomb / Grafana** | Their OTLP endpoint plus the vendor key in `OTEL_EXPORTER_OTLP_HEADERS` |
| **Anything else** | Its OTLP endpoint + headers. If it wants an SDK instead, add it in `app/tracing.py` — that file is the only place tracing is configured. |

`OTEL_EXPORTER_OTLP_HEADERS` is comma-separated `key=value`; percent-encode spaces
(`Bearer%20xyz`). The app appends `/v1/traces` to the endpoint, so give the base URL.

No endpoint handy? `TRACE_EXPORTER=console` prints full spans to stdout, which is enough to
see exactly what a product would receive.

---

## What a trace looks like

One user turn produces one trace:

```
chat.turn 1                        SERVER    session.id, user.id, gen_ai.conversation.id
└── chat claude-opus-5             CLIENT    request model, token usage, prompt + completion
    └── execute_tool wifi_channel_info   INTERNAL   tool name, call id, arguments, result
    └── chat claude-opus-5         CLIENT    the follow-up call that turns tool output into an answer
```

Attributes follow the OpenTelemetry GenAI semantic conventions. Because those conventions
changed mid-flight and products read different generations of them, the app emits **both**:

- `gen_ai.system` *and* `gen_ai.provider.name`
- message content as span **events** (`gen_ai.user.message`, `gen_ai.choice`, …) *and* as
  JSON **attributes** (`gen_ai.input.messages`, `gen_ai.output.messages`)

If your product renders conversations from one and not the other, that itself is a useful
finding. The names live in one file, `app/semconv.py`.

Set `CAPTURE_MESSAGE_CONTENT=false` to send the same spans with all prompt and response text
stripped — worth testing if the product will ever see production traffic.

---

## PRISMtrace

A second, independent tracer is wired in alongside OpenTelemetry, so you can watch
the same conversations land in both and compare. It is **entirely opt-in** — with
`PRISMTRACE_API_KEY` unset, every PRISMtrace code path is a no-op and nothing
changes.

```bash
# in .env (gitignored) — never in .env.example
PRISMTRACE_API_KEY=pt-sk-...
PRISMTRACE_PROJECT_ID=<project uuid>
PRISMTRACE_HOST=https://prismtrace-staging.up.railway.app
```

Per user turn, both backends post one trace to `/api/traces` and its span tree to
`/api/spans/ingest` — an `llm` span per model call and a `tool` span per tool
execution, all sharing one client-minted `trace_id`. The `langchain` backend
*additionally* attaches a `PRISMtraceCallbackHandler`, so you can see what their
native integration captures next to what explicit instrumentation captures.

Tool spans are posted explicitly on purpose: this app runs its own tool loop, so
tools are never invoked through LangChain and the callback handler never sees
them.

`ChatSession.session_id` is passed through as PRISMtrace's `session_id`, which is
what groups a conversation's turns into a trajectory. Handlers are cached per
session because the handler takes `session_id` at construction — one shared
handler would merge every conversation into a single trajectory.

Check what is active at any time:

```bash
curl -s localhost:8000/healthz | python3 -m json.tool
```

Config details and the three easy-to-hit gotchas are in [CLAUDE.md](CLAUDE.md).

## Generating traces without typing

Six scripted conversations cover the shapes worth checking:

```bash
python scripts/run_scenarios.py --list
python scripts/run_scenarios.py              # run them all, print trace ids
python scripts/run_scenarios.py single_tool  # just one
```

| Scenario | What it exercises |
| --- | --- |
| `plain_qa` | Simplest possible trace — one turn, one model call |
| `single_tool` | One tool call, then a final answer |
| `parallel_tools` | Two tool calls in one turn |
| `tool_error_recovery` | Tool raises, model gets an error result and recovers |
| `multi_turn` | Four turns in one session — session grouping, growing context |
| `injected_error` | A turn that raises before reaching the model |

The web UI has a **Run all scenarios** button that does the same thing.

---

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Effective config — backend, model, exporter, endpoint |
| `POST /chat` | `{message, session_id?, user_id?, backend?}` → reply **+ trace id** |
| `GET /scenarios` | List scripted scenarios |
| `POST /scenarios/run?name=` | Run one scenario, or all if `name` is omitted |
| `GET /` | Minimal chat UI |

`POST /chat` returns `session_id`; send it back to continue the same conversation. Omit it
to start a new one. The `backend` field lets you switch SDK per request, which is the
fastest way to compare the two paths in one tracing UI.

---

## Layout

```
app/
  config.py        env-driven settings, system prompt
  tracing.py       the only place tracing is configured
  semconv.py       GenAI attribute names
  messages.py      backend-neutral conversation model
  tools.py         two deterministic tools (one raises, on purpose)
  chat.py          turn span + tool-call loop
  backends/
    base.py              shared span recording — keeps both paths identical
    anthropic_backend.py direct SDK
    langchain_backend.py ChatAnthropic
  cli.py           terminal REPL
  server.py        FastAPI + web UI
scripts/
  selftest.py      offline checks, no API key
  run_scenarios.py scripted trace generator
```

---

## Notes and limits

- **Model:** defaults to `claude-opus-5` (`CHAT_MODEL` to change). The `thinking` parameter
  is deliberately not set — on Opus 5 that means adaptive thinking, the recommended default.
  `max_tokens` covers thinking *and* response text, hence the 8192 default.
- **Refusals:** Claude Opus 5 can decline a request with HTTP 200 and
  `stop_reason: "refusal"`. That is recorded as a span error rather than passing as a short
  answer, so refusals are visible in the tracing product.
- **Sessions are in memory.** Restarting the server drops history. Fine for a test harness,
  not a template for production.
- **No auth on the HTTP API.** Run it locally or behind something; do not expose it.
- **Tool loop caps at 6 rounds** per turn (`MAX_TOOL_ITERATIONS` in `app/chat.py`); hitting
  the cap marks the turn span as an error.
- **Manual instrumentation, by choice.** Auto-instrumentation libraries would hide exactly
  the thing under test. Every span here is written out in `app/chat.py` and
  `app/backends/base.py`, so what the product receives is inspectable.

## Licence

MIT — see [LICENSE](LICENSE).
