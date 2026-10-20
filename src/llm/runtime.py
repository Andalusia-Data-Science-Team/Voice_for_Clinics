"""Process-local shared AsyncOpenAI runtime for Azure AI Foundry and resilience."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

import httpx
from openai import AsyncOpenAI
from starlette.requests import Request

from core.config import Config
from llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
from llm.limiter import ModelLimiter
from llm.retry import is_insufficient_quota, is_retryable, retry_delay, retry_after_seconds, run_with_retries, status_code

logger = logging.getLogger(__name__)


def _build_foundry_client(timeout: httpx.Timeout, limits: httpx.Limits) -> tuple[AsyncOpenAI, str]:
    """Build an AsyncOpenAI client pointed at Azure AI Foundry.

    Returns (client, auth_method) where auth_method is 'managed_identity' or 'api_key'.
    """
    endpoint = Config.ai_foundry_endpoint()
    if not endpoint:
        raise RuntimeError(
            "LLM_PROVIDER=foundry but AI_FOUNDRY_ENDPOINT is not set. "
            "Set AI_FOUNDRY_ENDPOINT to your Azure AI Foundry resource endpoint."
        )

    endpoint_url = endpoint.rstrip("/")
    # Normalize to /openai/v1 base URL for the SDK
    if not endpoint_url.endswith("/openai/v1") and not endpoint_url.endswith("/openai/deployments"):
        base_url = f"{endpoint_url}/openai/v1"
    else:
        base_url = endpoint_url

    auth_method = Config.ai_foundry_auth_method()
    api_key = Config.ai_foundry_key()
    token_provider = None

    if auth_method == "managed_identity" or (not api_key and auth_method != "api_key"):
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # type: ignore
            credential = DefaultAzureCredential(logging_enable=False)
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )
            auth_method = "managed_identity"
            logger.info("[LLM] Foundry auth: Managed Identity (DefaultAzureCredential)")
        except Exception as exc:
            logger.debug("[LLM] DefaultAzureCredential not available (%s), trying API key", exc)

    if auth_method != "managed_identity" or token_provider is None:
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=foundry: neither DefaultAzureCredential succeeded "
                "nor AI_FOUNDRY_KEY / AZURE_API_KEY is configured."
            )
        auth_method = "api_key"
        logger.info("[LLM] Foundry auth: API key")

    http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    if auth_method == "managed_identity" and token_provider is not None:
        client = AsyncOpenAI(
            base_url=base_url,
            azure_ad_token_provider=token_provider,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
    else:
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    logger.info("[LLM] initialized Azure AI Foundry client base_url=%s auth=%s", base_url, auth_method)
    return client, auth_method


class LLMRuntime:
    """Own one AsyncOpenAI connection pool, model limiters, and circuit breakers per process for Azure AI Foundry."""

    def __init__(self, circuit_breaker: CircuitBreaker | None = None) -> None:
        timeout = httpx.Timeout(Config.LLM_TIMEOUT_S, connect=Config.LLM_CONNECT_TIMEOUT_S)
        limits = httpx.Limits(
            max_connections=Config.LLM_MAX_CONNECTIONS,
            max_keepalive_connections=Config.LLM_MAX_KEEPALIVE_CONNECTIONS,
        )
        self.client, self._auth_method = _build_foundry_client(timeout, limits)
        self.limiter = ModelLimiter()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5,
            window_s=30.0,
            recovery_timeout_s=15.0,
            name="FoundryLLMCircuitBreaker",
        )

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
        """Run a completion on Azure AI Foundry with circuit breaker, backoff, jitter, and adaptive concurrency."""
        # 1. Check Circuit Breaker
        await self.circuit_breaker.check_state()

        async def attempt() -> Any:
            await self.limiter.acquire(model)
            try:
                # Remove 'stream' if callers included it; complete() is non-streaming.
                create_kwargs.pop("stream", None)
                resp = await self.client.chat.completions.create(
                    **create_kwargs,
                    extra_headers=self._headers_for(key),
                )
                await self.limiter.record_feedback(model, status_code=200)
                await self.circuit_breaker.record_success()
                return resp
            except BaseException as exc:
                code = status_code(exc)
                is_429 = code == 429
                await self.limiter.record_feedback(model, status_code=code, is_rate_limited=is_429)
                if not is_insufficient_quota(exc):
                    await self.circuit_breaker.record_failure(exc)
                raise
            finally:
                await self.limiter.release(model)

        deadline = asyncio.get_running_loop().time() + Config.LLM_TOTAL_TIMEOUT_S
        sleep = self._deadline_sleep(deadline)
        async with asyncio.timeout(Config.LLM_TOTAL_TIMEOUT_S):
            return await run_with_retries(attempt, model=model, sleep=sleep)

    async def stream(
        self,
        *,
        model: str,
        create_kwargs: dict[str, Any],
        key: str | None = None,
    ) -> AsyncIterator[Any]:
        """Yield an async stream from Azure AI Foundry with circuit breaker, concurrency limits and retries."""
        # 1. Check Circuit Breaker
        await self.circuit_breaker.check_state()

        max_attempts = min(4, max(1, Config.LLM_MAX_ATTEMPTS))
        deadline = asyncio.get_running_loop().time() + Config.LLM_TOTAL_TIMEOUT_S
        sleep = self._deadline_sleep(deadline)
        timeout_scope = asyncio.timeout(Config.LLM_TOTAL_TIMEOUT_S)
        attempt_number = 0

        async with timeout_scope:
            while attempt_number < max_attempts:
                attempt_number += 1
                stream = None
                emitted = False
                retry_error: Exception | None = None
                await self.limiter.acquire(model)
                try:
                    # Remove 'stream' from kwargs; we pass it explicitly below.
                    create_kwargs.pop("stream", None)
                    stream = await self.client.chat.completions.create(
                        **create_kwargs,
                        extra_headers=self._headers_for(key),
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

                    await self.limiter.record_feedback(model, status_code=200)
                    await self.circuit_breaker.record_success()
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    code = status_code(exc)
                    is_429 = code == 429
                    await self.limiter.record_feedback(model, status_code=code, is_rate_limited=is_429)
                    if not is_insufficient_quota(exc):
                        await self.circuit_breaker.record_failure(exc)
                    if emitted or attempt_number >= max_attempts or not is_retryable(exc):
                        raise
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

    @staticmethod
    def _deadline_sleep(deadline: float) -> Callable[[float], Awaitable[None]]:
        loop = asyncio.get_running_loop()

        async def sleep(delay: float) -> None:
            remaining = max(0.0, deadline - loop.time())
            await asyncio.sleep(min(delay, remaining))

        return sleep

    def _headers_for(self, key: str | None = None) -> dict[str, str] | None:
        """Return headers required by Azure AI Foundry."""
        if self._auth_method == "api_key":
            headers: dict[str, str] = {}
            api_ver = Config.ai_foundry_api_version()
            if api_ver:
                headers["api-version"] = api_ver
            return headers or None
        return None


def get_llm_runtime(request: Request) -> LLMRuntime:
    """Return the shared runtime attached to a FastAPI request."""
    return request.app.state.llm
