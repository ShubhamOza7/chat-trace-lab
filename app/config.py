"""Environment-driven configuration.

Everything the app needs is read once, here, so the rest of the code never
touches os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    # Model
    backend: str = field(default_factory=lambda: os.getenv("CHAT_BACKEND", "anthropic").strip().lower())
    model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "claude-opus-5").strip())
    max_tokens: int = field(default_factory=lambda: _int("CHAT_MAX_TOKENS", 8192))

    # Tracing
    exporter: str = field(default_factory=lambda: os.getenv("TRACE_EXPORTER", "both").strip().lower())
    service_name: str = field(default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "chat-trace-lab").strip())
    otlp_endpoint: str = field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())
    capture_content: bool = field(default_factory=lambda: _bool("CAPTURE_MESSAGE_CONTENT", True))

    # Optional Langfuse convenience
    langfuse_public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "").strip())
    langfuse_secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "").strip())

    def validate(self) -> None:
        if self.backend not in {"anthropic", "langchain"}:
            raise ValueError(f"CHAT_BACKEND must be 'anthropic' or 'langchain', got {self.backend!r}")
        if self.exporter not in {"console", "otlp", "both", "none"}:
            raise ValueError(f"TRACE_EXPORTER must be console|otlp|both|none, got {self.exporter!r}")
        if self.exporter in {"otlp", "both"} and not self.otlp_endpoint:
            raise ValueError(
                "TRACE_EXPORTER includes OTLP but OTEL_EXPORTER_OTLP_ENDPOINT is unset. "
                "Set the endpoint, or use TRACE_EXPORTER=console."
            )


SETTINGS = Settings()

SYSTEM_PROMPT = (
    "You are the assistant inside a test harness whose only job is to produce realistic, "
    "traceable chat conversations. Answer the user's question directly and keep responses "
    "to the length the question needs. When a question involves a Wi-Fi channel number or a "
    "power level in dBm, use the provided tools rather than working it out yourself, so the "
    "trace shows a real tool call."
)
