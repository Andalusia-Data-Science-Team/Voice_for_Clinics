"""Adaptive per-model concurrency control for asynchronous LLM calls."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class LimiterStats:
    """Lightweight process-local metrics for one model limiter."""

    in_flight: int = 0
    acquire_count: int = 0
    total_wait_seconds: float = 0.0
    current_capacity: int = 8
    max_capacity: int = 8
    min_capacity: int = 1
    consecutive_successes: int = 0
    consecutive_rate_limits: int = 0


class DynamicSemaphore:
    """An asyncio semaphore that supports dynamic runtime resizing."""

    def __init__(self, initial_value: int) -> None:
        self._capacity = initial_value
        self._value = initial_value
        self._waiters: list[asyncio.Future[None]] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def value(self) -> int:
        return self._value

    def resize(self, new_capacity: int) -> None:
        """Resize capacity up or down safely."""
        delta = new_capacity - self._capacity
        self._capacity = new_capacity
        self._value += delta

        # Wake up waiting tasks if capacity increased
        while self._value > 0 and self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                self._value -= 1
                waiter.set_result(None)

    async def acquire(self) -> None:
        if self._value > 0:
            self._value -= 1
            return

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if not fut.done():
                self._waiters.remove(fut)
            else:
                self.release()
            raise

    def release(self) -> None:
        while self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                waiter.set_result(None)
                return
        self._value += 1


class ModelLimiter:
    """Maintain adaptive cancellation-safe semaphores per model slug.

    Supports dynamic rate tracking: automatically steps down concurrent workers
    when 429 response headers (rate limit / retry-after) are received, and
    gradually steps capacity back up as consecutive requests succeed.
    """

    def __init__(self, default_limit: int | None = None) -> None:
        self._default_limit = default_limit if default_limit is not None else Config.LLM_LIMIT_DEFAULT
        if self._default_limit < 1:
            raise ValueError("default_limit must be at least 1")
        self._semaphores: dict[str, DynamicSemaphore] = {}
        self._stats: dict[str, LimiterStats] = {}
        self._lock = asyncio.Lock()

    async def _get(self, model: str) -> tuple[DynamicSemaphore, LimiterStats]:
        async with self._lock:
            semaphore = self._semaphores.get(model)
            if semaphore is None:
                limit = Config.llm_model_limit(model) if self._default_limit == Config.LLM_LIMIT_DEFAULT else self._default_limit
                semaphore = DynamicSemaphore(limit)
                self._semaphores[model] = semaphore
                self._stats[model] = LimiterStats(
                    current_capacity=limit,
                    max_capacity=limit,
                    min_capacity=1,
                )
            return semaphore, self._stats[model]

    async def acquire(self, model: str) -> None:
        """Acquire a model slot and record wait/in-flight metrics."""
        semaphore, stats = await self._get(model)
        started = time.perf_counter()
        await semaphore.acquire()
        waited = time.perf_counter() - started
        stats.acquire_count += 1
        stats.total_wait_seconds += waited
        stats.in_flight += 1
        logger.info(
            "[LLM] limiter acquired model=%s in_flight=%d cap=%d wait_sec=%.3f",
            model, stats.in_flight, stats.current_capacity, waited,
        )

    async def release(self, model: str) -> None:
        """Release a model slot and record the resulting in-flight count."""
        semaphore, stats = await self._get(model)
        if stats.in_flight > 0:
            stats.in_flight -= 1
        semaphore.release()
        logger.info("[LLM] limiter released model=%s in_flight=%d", model, stats.in_flight)

    async def record_feedback(self, model: str, *, status_code: int | None = None, is_rate_limited: bool = False) -> None:
        """Adjust concurrency limits adaptively based on response feedback.

        On 429: Steps down capacity (multiplicative decrease: halved or -1, min 1).
        On 2xx Success: Steps up capacity after sustained consecutive successes (additive increase).
        """
        semaphore, stats = await self._get(model)
        async with self._lock:
            if is_rate_limited or status_code == 429:
                stats.consecutive_rate_limits += 1
                stats.consecutive_successes = 0
                old_cap = stats.current_capacity
                new_cap = max(stats.min_capacity, old_cap // 2)
                if new_cap != old_cap:
                    stats.current_capacity = new_cap
                    semaphore.resize(new_cap)
                    logger.warning(
                        "[LLM] Adaptive concurrency step-down model=%s from=%d to=%d (429 rate limit)",
                        model, old_cap, new_cap,
                    )
            elif status_code is not None and 200 <= status_code < 300:
                stats.consecutive_successes += 1
                stats.consecutive_rate_limits = 0
                # After 5 consecutive successful requests, recover 1 slot up to max_capacity
                if stats.consecutive_successes >= 5 and stats.current_capacity < stats.max_capacity:
                    old_cap = stats.current_capacity
                    new_cap = min(stats.max_capacity, old_cap + 1)
                    stats.current_capacity = new_cap
                    stats.consecutive_successes = 0
                    semaphore.resize(new_cap)
                    logger.info(
                        "[LLM] Adaptive concurrency step-up model=%s from=%d to=%d (recovery)",
                        model, old_cap, new_cap,
                    )

    async def stats(self, model: str) -> LimiterStats:
        """Return a snapshot of metrics for a model slug."""
        _, stats = await self._get(model)
        return LimiterStats(**vars(stats))