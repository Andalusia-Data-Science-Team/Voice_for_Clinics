"""Single-hop model fallback, layered ABOVE the src/llm retry/timeout/limiter stack.

The retry layer (``src/llm/retry.py`` + ``LLMRuntime``) already retries a single
deployment on 429/5xx/timeout. This module adds a *different* behaviour: when a
primary deployment still fails after those retries are exhausted, make exactly
ONE call on a fallback deployment. A fallback is a new call on another
deployment, never a retry of the same one.

Trigger rules (owner-approved):
  * Eligible: 429, 5xx, timeouts, connection errors, deployment-not-found (404).
  * NOT eligible: 400, 401, 403, 422, ``insufficient_quota``, content-filter
    blocks, or schema-invalid output (the last two are reported as open
    questions, not silently retried).

At most ONE fallback hop is ever taken per call.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

import httpx
import openai

from llm.retry import is_insufficient_quota, status_code

logger = logging.getLogger(__name__)


def _is_content_filter(exc: BaseException) -> bool:
    """Return True when the provider rejected the request via its content filter."""
    code = getattr(exc, "code", None)
    if code == "content_filter":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict) and err.get("code") == "content_filter":
            return True
    return False


def is_fallback_eligible(exc: BaseException) -> bool:
    """Return whether a failed primary call may be retried once on a fallback.

    Eligible: 429, 5xx, timeouts, connection errors, deployment-not-found (404).
    Not eligible: 400, 401, 403, 422, insufficient_quota, content-filter blocks.
    """
    if is_insufficient_quota(exc):
        return False
    if _is_content_filter(exc):
        return False
    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.APIConnectionError,
            httpx.TimeoutException,
            httpx.ConnectError,
        ),
    ):
        return True
    code = status_code(exc)
    if code is None:
        return False
    return code == 429 or code == 404 or (500 <= code <= 599)


async def run_with_single_fallback(
    primary: Callable[[], Awaitable[Any]],
    fallback: Callable[[], Awaitable[Any]],
    *,
    primary_deployment: str,
    fallback_deployment: str,
    meta: Optional[dict] = None,
) -> Any:
    """Run ``primary``; on an eligible failure, run ``fallback`` exactly once.

    Records which deployment actually served the call and whether a fallback was
    used into ``meta`` (additive optional keys). Never loops: at most one hop.
    """
    try:
        result = await primary()
        if meta is not None:
            meta["served_deployment"] = primary_deployment
            meta["fallback_used"] = False
        return result
    except Exception as exc:
        if not is_fallback_eligible(exc):
            raise
        logger.warning(
            "[Fallback] primary deployment=%s failed (%s); trying fallback=%s once",
            primary_deployment,
            type(exc).__name__,
            fallback_deployment,
        )
        result = await fallback()
        if meta is not None:
            meta["served_deployment"] = fallback_deployment
            meta["fallback_used"] = True
        return result


__all__ = ["is_fallback_eligible", "run_with_single_fallback"]