"""Bounded asynchronous retry policy for OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
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
    """Return whether an API error indicates exhausted billing or quota.

    Detected strictly from error.code or error.type only.
    """
    code = getattr(exc, "code", None)
    err_type = getattr(exc, "type", None)
    if code == "insufficient_quota" or err_type == "insufficient_quota":
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if err.get("code") == "insufficient_quota" or err.get("type") == "insufficient_quota":
            return True

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get("error") if isinstance(data.get("error"), dict) else data
                if err.get("code") == "insufficient_quota" or err.get("type") == "insufficient_quota":
                    return True
        except Exception:
            pass
    return False


def is_retryable(exc: BaseException) -> bool:
    """Return whether an exception is safe to retry before output is emitted.

    Retries are limited to throttling (429), server failures (5xx) and *timeouts*
    only (tracker A1.3.1: "retry on 429 / 5xx / timeout only"). Plain connection
    errors such as ``openai.APIConnectionError`` are NOT retried — a dead or
    unreachable endpoint should not be retried in-place. Cross-deployment
    fail-over is handled separately, above this layer (``pipeline.fallback``).
    """
    if is_insufficient_quota(exc):
        return False
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
        return True
    code = status_code(exc)
    return code == 429 or (code is not None and 500 <= code <= 599)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Parse a Retry-After or retry-after-ms header containing seconds, ms, or an HTTP date."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None

    # Check millisecond header first (Azure AI Foundry standard)
    ms_val = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
    if ms_val:
        try:
            return max(0.0, min(float(ms_val) / 1000.0, Config.LLM_RETRY_AFTER_MAX_S))
        except (ValueError, TypeError):
            pass

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
    on_retry: Callable[[int, BaseException | None, float], Awaitable[None] | None] | None = None,
) -> T:
    """Run an async operation with bounded exponential retry and jitter.

    The operation is responsible for acquiring and releasing its model slot
    for each attempt, so retry waits never hold a semaphore slot.
    """
    sleeper = sleep or asyncio.sleep

    def before_sleep(retry_state) -> None:
        delay = retry_state.next_action.sleep if retry_state.next_action else 0.0
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "[LLM] retry model=%s attempt=%d status=%s wait_sec=%.3f",
            model,
            retry_state.attempt_number,
            status_code(exc) if exc else None,
            delay,
        )
        if on_retry is not None:
            cb_res = on_retry(retry_state.attempt_number, exc, delay)
            if asyncio.iscoroutine(cb_res):
                asyncio.create_task(cb_res)

    # Hard cap of 4 attempts (tracker A1.3.1: "max 4 attempts"), matching
    # LLMRuntime.stream(). LLM_MAX_ATTEMPTS may lower the cap but never raise it.
    max_attempts = min(4, max(1, Config.LLM_MAX_ATTEMPTS))
    retrying = AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=_wait_with_retry_after,
        retry=retry_if_exception(is_retryable),
        reraise=True,
        before_sleep=before_sleep,
    )
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
    """Combine tenacity full jitter with a provider Retry-After / retry-after-ms value."""
    # Full jitter: random between 0 and min(max_backoff, base * 2^(attempt - 1))
    delay = calculate_backoff_delay(retry_state.attempt_number)
    if retry_state.outcome is not None:
        exc = retry_state.outcome.exception()
        if exc is not None:
            delay = max(delay, retry_after_seconds(exc) or 0.0)
    return delay


def calculate_backoff_delay(
    attempt_number: int,
    exc: BaseException | None = None,
    base_s: float = 1.0,
    max_backoff_s: float | None = None,
) -> float:
    """Calculate full-jitter exponential backoff:
    t_wait = min(t_max, t_base * 2^(attempt - 1)) * uniform(0, 1)

    Combined with Retry-After / retry-after-ms header if provided.
    """
    max_backoff = max_backoff_s if max_backoff_s is not None else Config.LLM_BACKOFF_MAX_S
    high = min(max_backoff, base_s * (2 ** max(0, attempt_number - 1)))
    delay = random.uniform(0.0, high)
    if exc is not None:
        delay = max(delay, retry_after_seconds(exc) or 0.0)
    return delay


retry_delay = calculate_backoff_delay
