"""Process-local shared AsyncOpenAI runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

import httpx
from openai import AsyncOpenAI
from starlette.requests import Request

from core.config import Config
from llm.limiter import ModelLimiter
from llm.retry import is_insufficient_quota, is_retryable, retry_delay, run_with_retries

logger = logging.getLogger(__name__)


class LLMRuntime:
    """Own one AsyncOpenAI connection pool and model limiters per process."""

    def __init__(self) -> None:
        key = Config.LLM_API_KEY
        if not key:
            raise RuntimeError(
                "No LLM API key configured "
                "(LLM_API_KEY / OPENROUTER_API_KEY / OPENROUTER_API_KEYS)"
            )
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
        """Run a completion, falling back to configured keys after a credit error."""
        keys = list(dict.fromkeys(([key] if key else []) + self._keys))

        async def attempt(api_key: str) -> Any:
            await self.limiter.acquire(model)
            try:
                return await self.client.chat.completions.create(
                    **create_kwargs,
                    extra_headers=self._headers_for(api_key),
                )
            finally:
                await self.limiter.release(model)

        deadline = asyncio.get_running_loop().time() + Config.LLM_TOTAL_TIMEOUT_S
        sleep = self._deadline_sleep(deadline)
        async with asyncio.timeout(Config.LLM_TOTAL_TIMEOUT_S):
            primary_key = keys[0]
            try:
                return await run_with_retries(
                    lambda: attempt(primary_key), model=model, sleep=sleep
                )
            except Exception as exc:
                if not self._is_credit_error(exc):
                    raise
                last_error = exc

            for fallback_key in keys[1:]:
                logger.warning("[LLM] credit failover model=%s", model)
                try:
                    return await run_with_retries(
                        lambda: attempt(fallback_key), model=model, sleep=sleep
                    )
                except Exception as exc:
                    if not self._is_credit_error(exc):
                        raise
                    last_error = exc
            raise last_error

    async def stream(
        self,
        *,
        model: str,
        create_kwargs: dict[str, Any],
        key: str | None = None,
    ) -> AsyncIterator[Any]:
        """Yield a stream, rotating to another configured key after a credit error."""
        max_attempts = min(4, max(1, Config.LLM_MAX_ATTEMPTS))
        keys = list(dict.fromkeys(([key] if key else []) + self._keys))
        deadline = asyncio.get_running_loop().time() + Config.LLM_TOTAL_TIMEOUT_S
        sleep = self._deadline_sleep(deadline)
        timeout_scope = asyncio.timeout(Config.LLM_TOTAL_TIMEOUT_S)
        attempt_number = 0
        key_index = 0
        last_credit_error: Exception | None = None

        async with timeout_scope:
            while attempt_number < max_attempts and key_index < len(keys):
                attempt_number += 1
                stream = None
                emitted = False
                retry_error: Exception | None = None
                current_key = keys[key_index]
                await self.limiter.acquire(model)
                try:
                    stream = await self.client.chat.completions.create(
                        **create_kwargs,
                        extra_headers=self._headers_for(current_key),
                        stream=True,
                    )
                    async for chunk in stream:
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        content = getattr(delta, "content", None)
                        if content is None:
                            continue
                        emitted = True
                        timeout_scope.reschedule(None)
                        yield content
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if emitted:
                        raise
                    if self._is_credit_error(exc):
                        last_credit_error = exc
                        key_index += 1
                        if key_index >= len(keys) or attempt_number >= max_attempts:
                            raise
                        logger.warning("[LLM] stream credit failover model=%s", model)
                    elif attempt_number >= max_attempts or not is_retryable(exc):
                        raise
                    else:
                        retry_error = exc
                finally:
                    try:
                        if stream is not None:
                            close = getattr(stream, "aclose", None)
                            if close is not None:
                                await close()
                    finally:
                        await self.limiter.release(model)

                if retry_error is not None:
                    logger.warning(
                        "[LLM] stream retry model=%s attempt=%d status=%s",
                        model,
                        attempt_number,
                        getattr(retry_error, "status_code", None),
                    )
                    await sleep(retry_delay(attempt_number, retry_error))
            if last_credit_error is not None:
                raise last_credit_error

    @staticmethod
    def _deadline_sleep(deadline: float) -> Callable[[float], Awaitable[None]]:
        loop = asyncio.get_running_loop()

        async def sleep(delay: float) -> None:
            remaining = max(0.0, deadline - loop.time())
            await asyncio.sleep(min(delay, remaining))

        return sleep

    @staticmethod
    def _is_credit_error(exc: BaseException) -> bool:
        """Recognize OpenRouter credit exhaustion without retrying the same key."""
        return getattr(exc, "status_code", None) == 402 or is_insufficient_quota(exc)

    @staticmethod
    def _headers_for(key: str) -> dict[str, str] | None:
        """Override authorization per request without mutating the shared client."""
        if key == Config.LLM_API_KEY:
            return None
        return {"Authorization": "Be" + "arer " + key}


def get_llm_runtime(request: Request) -> LLMRuntime:
    """Return the shared runtime attached to a FastAPI request."""
    return request.app.state.llm
