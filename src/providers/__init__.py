"""
src/providers/__init__.py — M2.1: Provider package public API

This module is the single import surface for the entire provider layer.
Consumers should always import from here, never from sub-modules directly:

    from providers import LLMProvider, STTProvider, ProviderHealthResult
    from providers import get_llm, get_stt          # Phase 3 factory singletons

Phase 1 exports:  LLMProvider, STTProvider, ProviderHealthResult
Phase 2 exports:  FoundryLLMProvider, MunsitSTTProvider, VoxtralSTTProvider
Phase 3 exports:  get_llm, get_stt (lazy-imported to avoid circular deps)
"""
from __future__ import annotations

from .base import (
    LLMProvider,
    ProviderHealthResult,
    STTProvider,
)
from .foundry_llm import FoundryLLMProvider
from .munsit_stt import MunsitSTTProvider
from .voxtral_stt import VoxtralSTTProvider

__all__ = [
    # ── Phase 1: contracts ────────────────────────────────────────────────
    "LLMProvider",
    "STTProvider",
    "ProviderHealthResult",
    # ── Phase 2: concrete adapters ────────────────────────────────────────
    "FoundryLLMProvider",
    "MunsitSTTProvider",
    "VoxtralSTTProvider",
    # ── Phase 3: factory singletons & builders ───────────────────────────
    "build_llm_provider",
    "build_stt_provider",
    "get_llm",
    "get_stt",
]

def __getattr__(name: str):
    """Lazy-load factory functions to avoid circular import issues."""
    if name in ("build_llm_provider", "build_stt_provider", "get_llm", "get_stt", "reset_providers"):
        from . import factory
        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
