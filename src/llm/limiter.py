"""Per-model concurrency control for asynchronous LLM calls."""

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


class ModelLimiter:
    """Maintain one cancellation-safe semaphore for each model slug.

    Limits are process-local. Size them with Little's law:
    ``in_flight ≈ calls/s × latency``. For example, 30 encounters/minute,
    two calls each, and eight seconds of latency imply about eight slots.
    """

    def __init__(self, default_limit: int | None = None) -> None:
        self._default_limit = default_limit if default_limit is not None else Config.LLM_LIMIT_DEFAULT
        if self._default_limit < 1:
            raise ValueError("default_limit must be at least 1")
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._stats: dict[str, LimiterStats] = {}
        self._lock = asyncio.Lock()

    async def _get(self, model: str) -> tuple[asyncio.Semaphore, LimiterStats]:
        async with self._lock:
            semaphore = self._semaphores.get(model)
            if semaphore is None:
                limit = Config.llm_model_limit(model) if self._default_limit == Config.LLM_LIMIT_DEFAULT else self._default_limit
                semaphore = asyncio.Semaphore(limit)
                self._semaphores[model] = semaphore
                self._stats[model] = LimiterStats()
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
            "[LLM] limiter acquired model=%s in_flight=%d wait_sec=%.3f",
            model, stats.in_flight, waited,
        )

    async def release(self, model: str) -> None:
        """Release a model slot and record the resulting in-flight count."""
        semaphore, stats = await self._get(model)
        if stats.in_flight > 0:
            stats.in_flight -= 1
        semaphore.release()
        logger.info("[LLM] limiter released model=%s in_flight=%d", model, stats.in_flight)

    async def stats(self, model: str) -> LimiterStats:
        """Return a snapshot of metrics for a model slug."""
        _, stats = await self._get(model)
        return LimiterStats(**vars(stats))