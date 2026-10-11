"""Bounded asynchronous retry policy for OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import email.utils
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, TypeVar

import httpx
import openai
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from core.config import Config

T = TypeVar("T")
logger = logging.getLogger(__name__)


def status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from an SDK or HTTPX exception."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status


def is_insufficient_quota(exc: BaseException) -> bool:
    """Return whether an API error indicates exhausted billing or quota."""
    text = str(exc).lower()
    code = getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            text += f" {response.json()}".lower()
        except Exception:
            pass
    return "insufficient_quota" in text or "billing" in text or code == "insufficient_quota"


def is_retryable(exc: BaseException) -> bool:
    """Return whether an exception is safe to retry before output is emitted."""
    if is_insufficient_quota(exc):
        return False
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, httpx.TimeoutException)):
        return True
    code = status_code(exc)
    return code == 429 or (code is not None and 500 <= code <= 599)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Parse a Retry-After header containing seconds or an HTTP date."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, min(float(value), Config.LLM_RETRY_AFTER_MAX_S))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, min(delay, Config.LLM_RETRY_AFTER_MAX_S))
        except (TypeError, ValueError, OverflowError):
            return None


async def run_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    model: str,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Run an async operation with bounded exponential retry and jitter.

    The operation is responsible for acquiring and releasing its model slot
    for each attempt, so retry waits never hold a semaphore slot.
    """
    sleeper = sleep or asyncio.sleep
    def before_sleep(retry_state) -> None:
        delay = retry_state.next_action.sleep if retry_state.next_action else 0.0
        logger.warning(
            "[LLM] retry model=%s attempt=%d status=%s wait_sec=%.3f",
            model,
            retry_state.attempt_number,
            status_code(retry_state.outcome.exception()) if retry_state.outcome else None,
            delay,
        )

    retrying = AsyncRetrying(
        stop=stop_after_attempt(max(1, Config.LLM_MAX_ATTEMPTS)),
        wait=wait_random_exponential(multiplier=1, max=Config.LLM_BACKOFF_MAX_S),
        retry=retry_if_exception(is_retryable),
        reraise=True,
        before_sleep=before_sleep,
    )
    retrying.wait = _wait_with_retry_after
    retrying.sleep = sleeper
    async for attempt in retrying:
        with attempt:
            try:
                return await operation()
            except BaseException as exc:
                if not is_retryable(exc):
                    raise
                raise
        # AsyncRetrying performs the calculated wait before the next attempt.
    raise RuntimeError("retry loop exited unexpectedly")


def _wait_with_retry_after(retry_state) -> float:
    """Combine tenacity jitter with a provider Retry-After value."""
    delay = wait_random_exponential(
        multiplier=1,
        max=Config.LLM_BACKOFF_MAX_S,
    )(retry_state)
    if retry_state.outcome is not None:
        delay = max(delay, retry_after_seconds(retry_state.outcome.exception()) or 0.0)
    return delay
