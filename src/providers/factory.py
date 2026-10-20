"""
src/providers/factory.py — M2.2 Phase 3: Provider Factory & Singleton Registry

Provides zero-code switching for LLM and STT providers via environment variables
(LLM_PROVIDER, STT_PROVIDER). Supports thread-safe/asyncio-safe singleton resolution
and standalone creation for testing or isolated runtime contexts.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from core.config import Config
from providers.base import LLMProvider, STTProvider

logger = logging.getLogger(__name__)

_llm_instance: Optional[LLMProvider] = None
_stt_instance: Optional[STTProvider] = None


def build_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Instantiate the configured LLM provider.

    Args:
        provider_name: Optional override. Defaults to LLM_PROVIDER env / Config.

    Returns:
        LLMProvider: Concrete provider instance (e.g. FoundryLLMProvider).

    Raises:
        ValueError: If provider_name is unknown.
    """
    selected = (provider_name or os.getenv("LLM_PROVIDER", getattr(Config, "LLM_PROVIDER", "foundry"))).strip().lower()

    if selected == "foundry":
        from llm.runtime import LLMRuntime
        from providers.foundry_llm import FoundryLLMProvider

        logger.info("[Factory] Instantiating FoundryLLMProvider with shared LLMRuntime")
        return FoundryLLMProvider(LLMRuntime())

    elif selected == "openrouter":
        from providers.openrouter_llm import OpenRouterLLMProvider

        logger.info("[Factory] Instantiating OpenRouterLLMProvider")
        return OpenRouterLLMProvider()

    raise ValueError(f"Unknown LLM_PROVIDER: {selected!r}. Supported options: ['foundry', 'openrouter']")


def build_stt_provider(provider_name: Optional[str] = None) -> STTProvider:
    """Instantiate the configured STT provider.

    Args:
        provider_name: Optional override. Defaults to STT_PROVIDER env / Config.

    Returns:
        STTProvider: Concrete provider instance (e.g. MunsitSTTProvider, VoxtralSTTProvider).

    Raises:
        ValueError: If provider_name is unknown.
    """
    selected = (provider_name or os.getenv("STT_PROVIDER", getattr(Config, "STT_PROVIDER", "munsit"))).strip().lower()

    if selected == "munsit":
        from providers.munsit_stt import MunsitSTTProvider

        api_key = os.getenv("MUNSIT_API_KEY", getattr(Config, "MUNSIT_API_KEY", "") or "")
        logger.info("[Factory] Instantiating MunsitSTTProvider")
        return MunsitSTTProvider(api_key=api_key)

    elif selected == "voxtral":
        from providers.voxtral_stt import VoxtralSTTProvider

        api_key = os.getenv("MISTRAL_API_KEY", getattr(Config, "MISTRAL_API_KEY", "") or "")
        logger.info("[Factory] Instantiating VoxtralSTTProvider")
        return VoxtralSTTProvider(api_key=api_key)

    raise ValueError(f"Unknown STT_PROVIDER: {selected!r}. Supported options: ['munsit', 'voxtral']")


def get_llm() -> LLMProvider:
    """Return the process-wide LLMProvider singleton."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = build_llm_provider()
    return _llm_instance


def get_stt() -> STTProvider:
    """Return the process-wide STTProvider singleton."""
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = build_stt_provider()
    return _stt_instance


def reset_providers() -> None:
    """Reset singletons (primarily used for unit testing)."""
    global _llm_instance, _stt_instance
    _llm_instance = None
    _stt_instance = None


__all__ = [
    "build_llm_provider",
    "build_stt_provider",
    "get_llm",
    "get_stt",
    "reset_providers",
]
