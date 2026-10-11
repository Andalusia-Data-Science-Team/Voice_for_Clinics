"""Process-local shared AsyncOpenAI runtime.

Supports two providers, selected by LLM_PROVIDER env var at startup:
  - "openrouter" (default): existing behavior, zero changes to call path.
  - "foundry": Azure AI Foundry endpoint, API-key or managed-identity auth.

Auth for Foundry:
  Production:  azure-identity DefaultAzureCredential bearer token provider.
               The openai SDK accepts a callable token provider via the
               azure-openai style `azure_ad_token_provider` kwarg.
  Local dev:   Falls back to AI_FOUNDRY_KEY / AZURE_API_KEY if set.

SECURITY: never logs or exposes keys, prompts, transcripts, or PHI.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import os
import httpx
from openai import AsyncOpenAI
from starlette.requests import Request

from core.config import Config
from core.telemetry import track_llm_call
from llm.limiter import ModelLimiter
from llm.retry import (
    calculate_backoff_delay,
    is_insufficient_quota,
    is_retryable,
    retry_after_seconds,
    run_with_retries,
)

logger = logging.getLogger(__name__)


def _build_foundry_client(timeout: httpx.Timeout, limits: httpx.Limits) -> tuple[AsyncOpenAI, str]:
    """Build an AsyncOpenAI client pointed at Azure AI Foundry.

    Returns (client, auth_method) where auth_method is 'managed_identity' or 'api_key'.

    Auth priority:
      1. azure-identity DefaultAzureCredential (production/CI with managed identity)
      2. AI_FOUNDRY_KEY / AZURE_API_KEY (local dev fallback)
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

    auth_method = "api_key"
    api_key: str | None = None
    token_provider = None

    # Try managed identity first (azure-identity)
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # type: ignore
        credential = DefaultAzureCredential(logging_enable=False)
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
        auth_method = "managed_identity"
        logger.info("[LLM] Foundry: using managed identity (DefaultAzureCredential)")
    except ImportError:
        logger.info("[LLM] Foundry: azure-identity not installed, falling back to API key")
    except Exception as exc:
        logger.warning("[LLM] Foundry: managed identity init failed (%s), falling back to API key", type(exc).__name__)

    if auth_method == "api_key" or token_provider is None:
        # API-key fallback for local dev
        from core.azure_identity import get_secret  # local import — no circular issue
        api_key = get_secret("AI_FOUNDRY_KEY") or Config.ai_foundry_key() or ""
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=foundry: no auth available. "
                "Either configure azure-identity (managed identity) or set AI_FOUNDRY_KEY."
            )
        auth_method = "api_key"

    # Build the SDK client.
    # azure_ad_token_provider is only passed when managed identity is active.
    client_kwargs: dict[str, Any] = {
        "base_url": base_url,
        "max_retries": 0,
        "timeout": timeout,
        "http_client": httpx.AsyncClient(timeout=timeout, limits=limits),
    }

    if auth_method == "managed_identity" and token_provider is not None:
        # Use a dummy api_key value (SDK requires something non-empty);
        # the actual bearer token comes from the token_provider.
        client_kwargs["api_key"] = "MANAGED_IDENTITY"
        client_kwargs["azure_ad_token_provider"] = token_provider
    else:
        client_kwargs["api_key"] = api_key
        client_kwargs["default_headers"] = {"api-key": api_key}

    client = AsyncOpenAI(**client_kwargs)
    logger.info("[LLM] Initialized Azure AI Foundry backend: %s auth=%s", base_url, auth_method)
    return client, auth_method


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

        if llm_provider == "foundry":
            self.client, self._auth_method = _build_foundry_client(timeout, limits)
            self._provider = "foundry"
            self._primary_key = ""  # key managed externally or via token provider
        else:
            # Default: OpenRouter — behavior identical to before
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
            self._auth_method = "api_key"
            self._primary_key = key
            logger.info("[LLM] Initialized OpenRouter backend: %s", Config.LLM_BASE_URL)

    @property
    def provider(self) -> str:
        """Return the active provider name: 'foundry' or 'openrouter'."""
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

        async def _run() -> Any:
            async with track_llm_call(model=model, operation="complete") as ctx:
                res = await run_with_retries(attempt, model=model)
                usage = getattr(res, "usage", None)
                if usage:
                    ctx.tokens = getattr(usage, "total_tokens", None)
                return res

        total_timeout = Config.LLM_TOTAL_TIMEOUT_S
        if total_timeout and total_timeout > 0:
            return await asyncio.wait_for(_run(), timeout=total_timeout)
        return await _run()

    async def stream(
        self,
        *,
        model: str,
        create_kwargs: dict[str, Any],
        key: str | None = None,
    ) -> AsyncIterator[Any]:
        """Yield a completion stream without blocking the event loop.

        A failed stream is retried only before the first content chunk is yielded.
        The slot is released before backoff sleep and re-acquired on retry.
        Total timeout applies only until the first content chunk.
        """
        max_attempts = max(1, Config.LLM_MAX_ATTEMPTS)
        total_timeout = Config.LLM_TOTAL_TIMEOUT_S
        deadline = (time.monotonic() + total_timeout) if total_timeout and total_timeout > 0 else None

        async with track_llm_call(model=model, operation="stream"):
            for attempt_number in range(1, max_attempts + 1):
                if deadline is not None and time.monotonic() >= deadline:
                    raise asyncio.TimeoutError("LLM total timeout exceeded before first stream chunk")

                stream = None
                slot_held = False
                headers = self._headers_for(key)
                try:
                    remaining = max(0.001, deadline - time.monotonic()) if deadline is not None else None
                    if remaining is not None:
                        await asyncio.wait_for(self.limiter.acquire(model), timeout=remaining)
                    else:
                        await self.limiter.acquire(model)
                    slot_held = True

                    create_coro = self.client.chat.completions.create(
                        **create_kwargs,
                        extra_headers=headers,
                        stream=True,
                    )
                    if remaining is not None:
                        remaining = max(0.001, deadline - time.monotonic())
                        stream = await asyncio.wait_for(create_coro, timeout=remaining)
                    else:
                        stream = await create_coro

                    chunk_iter = stream.__aiter__()
                    first_content = None
                    while True:
                        if remaining is not None:
                            remaining = max(0.001, deadline - time.monotonic())
                            chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=remaining)
                        else:
                            chunk = await chunk_iter.__anext__()
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        content = getattr(getattr(choices[0], "delta", None), "content", None)
                        if content is not None:
                            first_content = content
                            break

                except StopAsyncIteration:
                    try:
                        if stream is not None:
                            close = getattr(stream, "aclose", None)
                            if close is not None:
                                await close()
                    finally:
                        if slot_held:
                            await self.limiter.release(model)
                    return

                except asyncio.CancelledError:
                    try:
                        if stream is not None:
                            close = getattr(stream, "aclose", None)
                            if close is not None:
                                await close()
                    finally:
                        if slot_held:
                            await self.limiter.release(model)
                    raise

                except Exception as exc:
                    try:
                        if stream is not None:
                            close = getattr(stream, "aclose", None)
                            if close is not None:
                                await close()
                    finally:
                        if slot_held:
                            await self.limiter.release(model)
                            slot_held = False

                    if attempt_number >= max_attempts or not is_retryable(exc):
                        raise

                    delay = calculate_backoff_delay(attempt_number, exc)
                    logger.warning(
                        "[LLM] stream retry model=%s attempt=%d status=%s wait_sec=%.3f",
                        model, attempt_number, getattr(exc, "status_code", None), delay,
                    )

                    if deadline is not None and (time.monotonic() + delay) >= deadline:
                        raise asyncio.TimeoutError("LLM total timeout exceeded during stream retry backoff")

                    await asyncio.sleep(delay)
                    continue

                try:
                    yield first_content
                    async for chunk in chunk_iter:
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        content = getattr(getattr(choices[0], "delta", None), "content", None)
                        if content is not None:
                            yield content
                    return
                finally:
                    try:
                        if stream is not None:
                            close = getattr(stream, "aclose", None)
                            if close is not None:
                                await close()
                    finally:
                        if slot_held:
                            await self.limiter.release(model)
                            slot_held = False

    def _headers_for(self, key: str | None) -> dict[str, str] | None:
        """Return a request-scoped authorization override without mutating the client."""
        if self._provider == "foundry":
            if self._auth_method == "api_key":
                # Add api-version header for Foundry API-key auth
                headers: dict[str, str] = {}
                api_ver = Config.ai_foundry_api_version()
                if api_ver:
                    headers["api-version"] = api_ver
                return headers or None
            # Managed identity: token provider handles auth; no extra headers needed
            return None

        # OpenRouter: per-request key override
        if key and key != self._primary_key:
            return {"Authorization": f"Bearer {key}"}
        return None


def get_llm_runtime(request: Request) -> LLMRuntime:
    """Return the shared runtime attached to a FastAPI request."""
    return request.app.state.llm
