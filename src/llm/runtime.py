"""Process-local shared AsyncOpenAI runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import os
import httpx
from openai import AsyncOpenAI
from starlette.requests import Request

from core.config import Config
from core.telemetry import track_llm_call
from llm.limiter import ModelLimiter
from llm.retry import is_insufficient_quota, is_retryable, retry_after_seconds, run_with_retries

logger = logging.getLogger(__name__)


class LLMRuntime:
    """Own one AsyncOpenAI connection pool and model limiters per process."""

    def __init__(self) -> None:
        timeout = httpx.Timeout(Config.LLM_TIMEOUT_S, connect=Config.LLM_CONNECT_TIMEOUT_S)
        limits = httpx.Limits(
            max_connections=Config.LLM_MAX_CONNECTIONS,
            max_keepalive_connections=Config.LLM_MAX_KEEPALIVE_CONNECTIONS,
        )
        self.limiter = ModelLimiter()

        # ── Determine backend: Azure AI Foundry or OpenRouter ──
        llm_provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
        foundry_endpoint = Config.ai_foundry_endpoint() if llm_provider == "azure" else None
        if foundry_endpoint:
            from core.azure_identity import get_secret
            api_key = get_secret("AI_FOUNDRY_KEY") or Config.ai_foundry_key() or ""
            if not api_key:
                raise RuntimeError("No AI Foundry API key configured (AI_FOUNDRY_KEY / AZURE_API_KEY)")
            
            # Normalize Azure OpenAI / Foundry base URL
            endpoint_url = foundry_endpoint.rstrip("/")
            if not endpoint_url.endswith("/openai/v1") and not endpoint_url.endswith("/openai/deployments"):
                base_url = f"{endpoint_url}/openai/v1"
            else:
                base_url = endpoint_url

            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={"api-key": api_key},
                timeout=timeout,
                max_retries=0,
                http_client=httpx.AsyncClient(timeout=timeout, limits=limits),
            )
            self._provider = "ai_foundry"
            self._primary_key = api_key
            self._keys = [api_key]
            logger.info("[LLM] Initialized Azure AI Foundry backend: %s", base_url)
        else:
            key = Config.llm_api_key()
            if not key:
                raise RuntimeError("No LLM API key configured (LLM_API_KEY / OPENROUTER_API_KEY)")
            self.client = AsyncOpenAI(
                base_url=Config.LLM_BASE_URL,
                api_key=key,
                timeout=timeout,
                max_retries=0,
                http_client=httpx.AsyncClient(timeout=timeout, limits=limits),
            )
            self._provider = "openrouter"
            self._primary_key = key
            self._keys = list(dict.fromkeys(Config.openrouter_api_keys() or [key]))
            logger.info("[LLM] Initialized OpenRouter backend: %s", Config.LLM_BASE_URL)

    @property
    def provider(self) -> str:
        """Return the active provider name: 'ai_foundry' or 'openrouter'."""
        return self._provider

    async def close(self) -> None:
        """Close the shared SDK client and its underlying HTTP transport."""
        await self.client.close()

    async def complete(
        self,
        *,
        model: str,
        create_kwargs: dict[str, Any],
        key: str | None = None,
    ) -> Any:
        """Run one non-streaming completion with limits and bounded retries."""
        async def attempt() -> Any:
            await self.limiter.acquire(model)
            try:
                headers = self._headers_for(key)
                return await self.client.chat.completions.create(
                    **create_kwargs,
                    extra_headers=headers,
                )
            finally:
                await self.limiter.release(model)

        async with track_llm_call(model=model, operation="complete") as ctx:
            try:
                res = await run_with_retries(attempt, model=model)
                usage = getattr(res, "usage", None)
                if usage:
                    ctx.tokens = getattr(usage, "total_tokens", None)
                return res
            except Exception as exc:
                if is_insufficient_quota(exc):
                    res = await self._complete_with_quota_rotation(
                        model=model, create_kwargs=create_kwargs, failed_key=key,
                    )
                    usage = getattr(res, "usage", None)
                    if usage:
                        ctx.tokens = getattr(usage, "total_tokens", None)
                    return res
                raise

    async def _complete_with_quota_rotation(
        self,
        *,
        model: str,
        create_kwargs: dict[str, Any],
        failed_key: str | None,
    ) -> Any:
        """Try remaining configured keys only after an explicit quota failure."""
        attempted = failed_key or Config.llm_api_key()
        for key in self._keys:
            if key == attempted:
                continue
            logger.warning("[LLM] quota rotation model=%s", model)

            async def attempt() -> Any:
                await self.limiter.acquire(model)
                try:
                    return await self.client.chat.completions.create(
                        **create_kwargs,
                        extra_headers=self._headers_for(key),
                    )
                finally:
                    await self.limiter.release(model)

            return await run_with_retries(attempt, model=model)
        raise RuntimeError("All configured LLM keys exhausted after a quota failure")

    async def stream(
        self,
        *,
        model: str,
        create_kwargs: dict[str, Any],
        key: str | None = None,
    ) -> AsyncIterator[Any]:
        """Yield a completion stream without blocking the event loop.

        A failed stream is retried only before the first chunk is yielded. The
        model slot remains held for the complete attempt, including cleanup.
        """
        max_attempts = max(1, Config.LLM_MAX_ATTEMPTS)
        async with track_llm_call(model=model, operation="stream"):
            for attempt_number in range(1, max_attempts + 1):
                stream = None
                emitted = False
                headers = self._headers_for(key)
                await self.limiter.acquire(model)
                try:
                    stream = await self.client.chat.completions.create(
                        **create_kwargs,
                        extra_headers=headers,
                        stream=True,
                    )
                    async for chunk in stream:
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        content = getattr(getattr(choices[0], "delta", None), "content", None)
                        if content is None:
                            continue
                        emitted = True
                        yield content
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if emitted or attempt_number >= max_attempts or not is_retryable(exc):
                        raise
                    logger.warning(
                        "[LLM] stream retry model=%s attempt=%d status=%s",
                        model, attempt_number, getattr(exc, "status_code", None),
                    )
                    await asyncio.sleep(max(0.0, retry_after_seconds(exc) or 0.0))
                finally:
                    if stream is not None:
                        close = getattr(stream, "aclose", None)
                        if close is not None:
                            await close()
                    await self.limiter.release(model)

    def _headers_for(self, key: str | None) -> dict[str, str] | None:
        """Return a request-scoped authorization override without mutating the client."""
        if self._provider == "ai_foundry":
            headers = {}
            if key and key != self._primary_key:
                headers["api-key"] = key
            # Add api-version query param / header support
            api_ver = Config.AI_FOUNDRY_API_VERSION
            if api_ver:
                headers["api-version"] = api_ver
            return headers or None

        if key and key != self._primary_key:
            return {"Authorization": f"Bearer {key}"}
        return None


def get_llm_runtime(request: Request) -> LLMRuntime:
    """Return the shared runtime attached to a FastAPI request."""
    return request.app.state.llm
