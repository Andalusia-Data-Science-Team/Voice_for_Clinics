"""
src/extractor.py — Dual-Client Extractor Adapter for Microsoft AI Foundry & OpenRouter

Provides seamless switching between Microsoft AI Foundry (Azure OpenAI / Model Inference)
and OpenRouter fallback, featuring exponential backoff with full jitter, adaptive
rate tracking, and circuit breaker protection.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from core.config import Config
from llm.runtime import LLMRuntime
from providers.base import LLMProvider
from providers.factory import build_llm_provider
from providers.foundry_llm import FoundryLLMProvider
from providers.openrouter_llm import OpenRouterLLMProvider

logger = logging.getLogger(__name__)


class DualClientExtractor:
    """Extensible client adapter for clinical note extraction and reasoning.

    Seamlessly switches between primary (Foundry) and secondary/fallback (OpenRouter)
    based on the `PROVIDER` or `LLM_PROVIDER` environment variable, or direct initialization.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        fallback_enabled: bool = True,
        runtime: Optional[LLMRuntime] = None,
    ) -> None:
        raw_provider = provider or os.getenv("PROVIDER") or os.getenv("LLM_PROVIDER") or getattr(Config, "LLM_PROVIDER", "foundry")
        self._provider_type = raw_provider.strip().lower()
        self._fallback_enabled = fallback_enabled

        self._runtime = runtime or LLMRuntime()
        self._foundry_client = FoundryLLMProvider(self._runtime)
        self._openrouter_client: Optional[OpenRouterLLMProvider] = None

        if self._provider_type == "openrouter":
            self._active_client: LLMProvider = self._get_openrouter_client()
        else:
            self._active_client = self._foundry_client

        logger.info(
            "[Extractor] Initialized DualClientExtractor active_provider=%s fallback_enabled=%s",
            self._provider_type,
            self._fallback_enabled,
        )

    def _get_openrouter_client(self) -> OpenRouterLLMProvider:
        if self._openrouter_client is None:
            self._openrouter_client = OpenRouterLLMProvider()
        return self._openrouter_client

    @property
    def active_provider_name(self) -> str:
        return self._active_client.provider_name

    @property
    def runtime(self) -> LLMRuntime:
        return self._runtime

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        """Execute a completion with automatic failover if configured."""
        try:
            return await self._active_client.complete(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            if not self._fallback_enabled:
                raise

            # If primary was foundry and failed, try OpenRouter fallback
            if self._active_client == self._foundry_client:
                logger.warning(
                    "[Extractor] Primary provider 'foundry' failed (%s); failing over to OpenRouter fallback",
                    exc,
                )
                fallback_client = self._get_openrouter_client()
                return await fallback_client.complete(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                )
            raise

    async def close(self) -> None:
        """Close clients and runtimes."""
        await self._foundry_client.close()
        if self._openrouter_client is not None:
            await self._openrouter_client.close()


# Alias for explicit task naming requirement
FoundryClient = FoundryLLMProvider
OpenRouterClient = OpenRouterLLMProvider
