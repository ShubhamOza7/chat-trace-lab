"""OpenTelemetry wiring.

One `init_tracing()` call sets up a TracerProvider that exports to any
OTLP/HTTP endpoint and/or the console. Nothing here is specific to a tracing
vendor — point `OTEL_EXPORTER_OTLP_ENDPOINT` at whichever product you are
evaluating.
"""

from __future__ import annotations

import base64
import logging
import os
import threading

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from .config import SETTINGS, Settings

log = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INITIALISED = False

INSTRUMENTATION_SCOPE = "chat-trace-lab"


def _apply_langfuse_headers(settings: Settings) -> None:
    """Build the Basic auth header Langfuse's OTLP endpoint expects.

    Purely a convenience for one common target. Any other product is configured
    the standard way, through OTEL_EXPORTER_OTLP_HEADERS.
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return
    if os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
        return  # explicit headers win
    token = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {token}"


def init_tracing(settings: Settings | None = None) -> None:
    """Install the global TracerProvider. Safe to call more than once."""
    global _INITIALISED
    settings = settings or SETTINGS

    with _INIT_LOCK:
        if _INITIALISED:
            return
        settings.validate()

        if settings.exporter == "none":
            _INITIALISED = True
            log.info("Tracing disabled (TRACE_EXPORTER=none)")
            return

        _apply_langfuse_headers(settings)

        resource = Resource.create(
            {
                "service.name": settings.service_name,
                "service.version": os.getenv("APP_VERSION", "0.1.0"),
                "deployment.environment.name": os.getenv("APP_ENV", "local"),
            }
        )
        provider = TracerProvider(resource=resource)

        if settings.exporter in {"otlp", "both"}:
            endpoint = settings.otlp_endpoint.rstrip("/") + "/v1/traces"
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            log.info("OTLP span exporter -> %s", endpoint)

        if settings.exporter in {"console", "both"}:
            # Simple (not batch) so spans print in the order they close, which
            # makes the console output readable while you are wiring things up.
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _INITIALISED = True


def tracer() -> trace.Tracer:
    return trace.get_tracer(INSTRUMENTATION_SCOPE)


def current_trace_id() -> str | None:
    """Hex trace id of the active span, for pasting into a tracing UI."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def shutdown_tracing() -> None:
    """Flush pending spans. Call before a short-lived process exits."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
