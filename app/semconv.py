"""OpenTelemetry GenAI semantic-convention attribute names.

These are spelled out as plain strings rather than imported from
`opentelemetry.semconv`, because the GenAI conventions are still moving and the
packaged constants lag behind what tracing vendors actually read.

Two generations of the convention are emitted side by side:

  * `gen_ai.system` (older) and `gen_ai.provider.name` (newer)
  * message content as span *events* (older) and as JSON *attributes* (newer)

Emitting both is deliberate: it maximises the chance that whichever product you
point this at renders the conversation without you editing any code.
"""

from __future__ import annotations

# --- Operation identity -----------------------------------------------------
OPERATION_NAME = "gen_ai.operation.name"
SYSTEM = "gen_ai.system"  # legacy spelling
PROVIDER_NAME = "gen_ai.provider.name"  # current spelling

# --- Request ----------------------------------------------------------------
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"

# --- Response ---------------------------------------------------------------
RESPONSE_ID = "gen_ai.response.id"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

# --- Usage ------------------------------------------------------------------
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ_TOKENS = "gen_ai.usage.cache_read_input_tokens"
USAGE_CACHE_WRITE_TOKENS = "gen_ai.usage.cache_creation_input_tokens"

# --- Conversation / content -------------------------------------------------
CONVERSATION_ID = "gen_ai.conversation.id"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"

# --- Tools ------------------------------------------------------------------
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_DESCRIPTION = "gen_ai.tool.description"

# --- Span events (legacy content carriers) ----------------------------------
EVENT_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_USER_MESSAGE = "gen_ai.user.message"
EVENT_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_TOOL_MESSAGE = "gen_ai.tool.message"
EVENT_CHOICE = "gen_ai.choice"

# --- App-specific attributes (not part of the spec) -------------------------
APP_BACKEND = "app.chat.backend"
APP_TURN_INDEX = "app.chat.turn_index"
APP_TOOL_ITERATIONS = "app.chat.tool_iterations"
APP_SCENARIO = "app.chat.scenario"
SESSION_ID = "session.id"
USER_ID = "user.id"
