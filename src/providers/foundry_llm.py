"""
src/providers/foundry_llm.py — M2.2: Concrete Azure AI Foundry LLM Provider Adapter

Wraps the existing LLMRuntime under the unified LLMProvider interface.
Delegates complete() and stream() to LLMRuntime, translating the generic
interface kwargs into Foundry-specific build_foundry_kwargs() calls via llm/provider.py.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

from llm.provider import build_foundry_kwargs
from llm.runtime import LLMRuntime
from providers.base import LLMProvider, ProviderHealthResult


class FoundryLLMProvider(LLMProvider):
    """Azure AI Foundry implementation of LLMProvider.

    Concurreny safe: reuses the underlying LLMRuntime connection pool,
    rate limiters, retry logic, and AsyncOpenAI client.
    """

    def __init__(self, runtime: LLMRuntime) -> None:
        if runtime is None:
            raise ValueError("runtime cannot be None")
        self._runtime = runtime

    @property
    def runtime(self) -> LLMRuntime:
        """Access underlying LLMRuntime if needed for diagnostics."""
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
        """Non-streaming chat completion via LLMRuntime."""
        kwargs = build_foundry_kwargs(
            model,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        return await self._runtime.complete(model=model, create_kwargs=kwargs)

    async def stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Async generator yielding token delta strings via LLMRuntime."""
        kwargs = build_foundry_kwargs(
            model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            response_format=response_format,
        )
        async for chunk in self._runtime.stream(model=model, create_kwargs=kwargs):
            yield chunk

    async def close(self) -> None:
        """Gracefully close the underlying runtime client."""
        await self._runtime.close()

    async def health_check(self) -> ProviderHealthResult:
        """Probe the Azure AI Foundry endpoint with a lightweight models.list() call.

        Applies a 5.0s internal timeout and never raises exceptions.
        """
        start = time.perf_counter()
        try:
            client = getattr(self._runtime, "client", None)
            if client is None:
                return ProviderHealthResult.unhealthy(
                    provider=self.provider_name,
                    detail="LLMRuntime client is uninitialized",
                )
            await asyncio.wait_for(client.models.list(), timeout=5.0)
            latency = round((time.perf_counter() - start) * 1000.0, 2)
            return ProviderHealthResult.healthy(provider=self.provider_name, latency_ms=latency)
        except Exception as exc:
            latency = round((time.perf_counter() - start) * 1000.0, 2)
            return ProviderHealthResult.unhealthy(
                provider=self.provider_name,
                detail=str(exc),
                latency_ms=latency,
            )


__all__ = ["FoundryLLMProvider"]
