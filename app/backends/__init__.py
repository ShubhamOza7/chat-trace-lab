"""Backend selection."""

from __future__ import annotations

from ..config import SETTINGS
from .base import ChatBackend


def build_backend(name: str | None = None) -> ChatBackend:
    name = (name or SETTINGS.backend).strip().lower()
    if name == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend()
    if name == "langchain":
        from .langchain_backend import LangChainBackend

        return LangChainBackend()
    raise ValueError(f"Unknown backend {name!r}; expected 'anthropic' or 'langchain'.")


__all__ = ["ChatBackend", "build_backend"]
