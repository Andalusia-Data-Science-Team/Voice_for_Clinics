"""Circuit Breaker for LLM service endpoints."""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"      # Normal operations; requests pass through
    OPEN = "OPEN"          # Failing; requests immediately rejected to protect backend
    HALF_OPEN = "HALF_OPEN"  # Testing recovery; allowing a trial probe request


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the circuit is OPEN."""
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class CircuitBreakerStats:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_state_change: float = field(default_factory=time.time)
    tripped_count: int = 0


class CircuitBreaker:
    """Threshold-based Circuit Breaker with sliding time window.

    - Consecutive failure threshold: e.g. 5 failures within window_s (default 30s) trips circuit.
    - Reset timeout: after recovery_timeout_s (default 15s), transitions from OPEN to HALF_OPEN.
    - In HALF_OPEN: a successful request closes the circuit; a failure immediately re-trips it OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_s: float = 30.0,
        recovery_timeout_s: float = 15.0,
        name: str = "LLMCircuitBreaker",
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.window_s = max(0.01, window_s)
        self.recovery_timeout_s = max(0.01, recovery_timeout_s)
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_timestamps: list[float] = []
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0
        self._last_state_change = time.time()
        self._tripped_count = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def check_state(self) -> None:
        """Verify circuit allows an execution. Trips or transitions if needed."""
        async with self._lock:
            now = time.time()
            if self._state == CircuitState.OPEN:
                elapsed = now - self._last_state_change
                if elapsed >= self.recovery_timeout_s:
                    self._state = CircuitState.HALF_OPEN
                    self._last_state_change = now
                    logger.info(
                        "[CircuitBreaker] %s transitioned OPEN -> HALF_OPEN (probing backend after %.1fs)",
                        self.name, elapsed,
                    )
                else:
                    retry_after = self.recovery_timeout_s - elapsed
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker {self.name} is OPEN. Retrying after {retry_after:.1f}s",
                        retry_after=retry_after,
                    )

    async def record_success(self) -> None:
        """Record successful execution."""
        async with self._lock:
            now = time.time()
            self._consecutive_successes += 1
            self._consecutive_failures = 0

            # Prune old timestamps
            cutoff = now - self.window_s
            self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._last_state_change = now
                logger.info("[CircuitBreaker] %s trial probe SUCCEEDED. Circuit transitioned HALF_OPEN -> CLOSED", self.name)

    async def record_failure(self, exc: Optional[BaseException] = None) -> None:
        """Record failed execution and trip circuit if threshold is reached."""
        # A cancellation (client disconnect / task teardown) is not a backend
        # failure and must never count toward tripping the circuit breaker.
        if isinstance(exc, asyncio.CancelledError):
            return
        async with self._lock:
            now = time.time()
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._failure_timestamps.append(now)

            # Keep only failures within window
            cutoff = now - self.window_s
            self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]

            recent_failures = len(self._failure_timestamps)

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_state_change = now
                self._tripped_count += 1
                logger.critical(
                    "[CircuitBreaker] %s trial probe FAILED. Circuit transitioned HALF_OPEN -> OPEN. Error: %s",
                    self.name, exc,
                )
            elif self._state == CircuitState.CLOSED and (
                self._consecutive_failures >= self.failure_threshold or recent_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._last_state_change = now
                self._tripped_count += 1
                logger.critical(
                    "[CircuitBreaker] %s TRIPPED: %d failures in past %.1fs (threshold=%d). Pausing worker pool. Error: %s",
                    self.name, recent_failures, self.window_s, self.failure_threshold, exc,
                )

    async def get_stats(self) -> CircuitBreakerStats:
        async with self._lock:
            return CircuitBreakerStats(
                state=self._state,
                failure_count=len(self._failure_timestamps),
                consecutive_failures=self._consecutive_failures,
                consecutive_successes=self._consecutive_successes,
                last_state_change=self._last_state_change,
                tripped_count=self._tripped_count,
            )
