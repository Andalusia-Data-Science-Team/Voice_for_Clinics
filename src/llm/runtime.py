"""Process-local shared AsyncOpenAI runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI
from starlette.requests import Request

from core.config import Config
from llm.limiter import ModelLimiter
from llm.retry import is_insufficient_quota, is_retryable, retry_after_seconds, run_with_retries

logger = logging.getLogger(__name__)


class LLMRuntime:
    """Own one AsyncOpenAI connection pool and model limiters per process."""

    def __init__(self) -> None:
        key = Config.LLM_API_KEY
        if not key:
            raise RuntimeError("No LLM API key configured (LLM_API_KEY / OPENROUTER_API_KEY)")
        timeout = httpx.Timeout(Config.LLM_TIMEOUT_S, connect=Config.LLM_CONNECT_TIMEOUT_S)
        limits = httpx.Limits(
            max_connections=Config.LLM_MAX_CONNECTIONS,
            max_keepalive_connections=Config.LLM_MAX_KEEPALIVE_CONNECTIONS,
        )
        self.client = AsyncOpenAI(
            base_url=Config.LLM_BASE_URL,
            api_key=key,
            timeout=timeout,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=timeout, limits=limits),
        )
        self.limiter = ModelLimiter()
        self._keys = list(dict.fromkeys(Config.OPENROUTER_API_KEYS or [key]))

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

        try:
            return await run_with_retries(attempt, model=model)
        except Exception as exc:
            if is_insufficient_quota(exc):
                return await self._complete_with_quota_rotation(
                    model=model, create_kwargs=create_kwargs, failed_key=key,
                )
            raise

    async def _complete_with_quota_rotation(
        self,
        *,
        model: str,
        create_kwargs: dict[str, Any],
        failed_key: str | None,
    ) -> Any:
        """Try remaining configured keys only after an explicit quota failure."""
        attempted = failed_key or Config.LLM_API_KEY
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

    @staticmethod
    def _headers_for(key: str | None) -> dict[str, str] | None:
        """Return a request-scoped authorization override without mutating the client."""
        if key and key != Config.LLM_API_KEY:
            return {"Authorization": f"Bearer {key}"}
        return None


def get_llm_runtime(request: Request) -> LLMRuntime:
    """Return the shared runtime attached to a FastAPI request."""
    return request.app.state.llm
