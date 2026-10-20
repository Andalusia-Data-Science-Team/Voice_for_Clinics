"""
src/providers/openrouter_llm.py — Dual-Client Fallback Provider Adapter for OpenRouter

Provides an AsyncOpenAI-compatible client pointing to OpenRouter API
(https://openrouter.ai/api/v1). Implements the unified LLMProvider interface
to allow seamless switching and fallback between Foundry and OpenRouter.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from core.config import Config
from providers.base import LLMProvider, ProviderHealthResult

logger = logging.getLogger(__name__)


class OpenRouterLLMProvider(LLMProvider):
    """OpenRouter implementation of LLMProvider.

    Supports OpenAI SDK chat completions format targeting OpenRouter's API endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 45.0,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self._timeout_s = timeout_s
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_s, connect=Config.LLM_CONNECT_TIMEOUT_S),
            limits=httpx.Limits(
                max_connections=Config.LLM_MAX_CONNECTIONS,
                max_keepalive_connections=Config.LLM_MAX_KEEPALIVE_CONNECTIONS,
            ),
        )
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key or "sk-dummy-key",
            timeout=self._timeout_s,
            max_retries=0,
            http_client=self._http_client,
        )

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        """Non-streaming chat completion via OpenRouter."""
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format:
            create_kwargs["response_format"] = response_format

        extra_headers = {
            "HTTP-Referer": "https://medvoice.andalusia.com",
            "X-Title": "MedVoice Clinical Assistant",
        }

        return await self._client.chat.completions.create(
            **create_kwargs,
            extra_headers=extra_headers,
        )

    async def stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Async generator yielding token deltas via OpenRouter."""
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if response_format:
            create_kwargs["response_format"] = response_format

        extra_headers = {
            "HTTP-Referer": "https://medvoice.andalusia.com",
            "X-Title": "MedVoice Clinical Assistant",
        }

        response = await self._client.chat.completions.create(
            **create_kwargs,
            extra_headers=extra_headers,
        )
        async for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content is not None:
                yield content

    async def close(self) -> None:
        """Close HTTP clients."""
        await self._client.close()
        await self._http_client.aclose()

    async def health_check(self) -> ProviderHealthResult:
        """Lightweight models.list() probe."""
        start = time.perf_counter()
        try:
            await asyncio.wait_for(self._client.models.list(), timeout=5.0)
            latency = round((time.perf_counter() - start) * 1000.0, 2)
            return ProviderHealthResult.healthy(provider=self.provider_name, latency_ms=latency)
        except Exception as exc:
            latency = round((time.perf_counter() - start) * 1000.0, 2)
            return ProviderHealthResult.unhealthy(
                provider=self.provider_name,
                detail=str(exc),
                latency_ms=latency,
            )


__all__ = ["OpenRouterLLMProvider"]
