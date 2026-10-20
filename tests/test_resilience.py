"""
Tests for Phase 3: Failure Mitigation & Microsoft Foundry Migration
- 3.1: Azure AI Foundry configuration & compliance
- 3.2: Dual-Client Adapter Pattern (Foundry vs OpenRouter fallback)
- 3.3: Resilience & Retry Architecture:
  - Exponential Backoff with full jitter formula
  - Adaptive Concurrency & dynamic rate stepping
  - Circuit Breaker threshold, trip state, and recovery
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
import openai

from core.config import Config
from extractor import DualClientExtractor
from llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException, CircuitState
from llm.limiter import DynamicSemaphore, ModelLimiter
from llm.retry import (
    calculate_backoff_delay,
    is_insufficient_quota,
    is_retryable,
    retry_after_seconds,
    run_with_retries,
)
from providers.factory import build_llm_provider
from providers.foundry_llm import FoundryLLMProvider
from providers.openrouter_llm import OpenRouterLLMProvider


class TestResilienceRetry:
    """Test 3.3 Exponential Backoff with Full Jitter and Retry-After headers."""

    def test_calculate_backoff_delay_bounds(self):
        # Attempt 1: 0 <= delay <= min(8.0, 1.0 * 2^0) = 1.0
        for _ in range(50):
            d = calculate_backoff_delay(attempt_number=1, base_s=1.0, max_backoff_s=8.0)
            assert 0.0 <= d <= 1.0

        # Attempt 3: 0 <= delay <= min(8.0, 1.0 * 2^2) = 4.0
        for _ in range(50):
            d = calculate_backoff_delay(attempt_number=3, base_s=1.0, max_backoff_s=8.0)
            assert 0.0 <= d <= 4.0

        # Attempt 5: 0 <= delay <= min(8.0, 1.0 * 2^4) = 8.0 (capped by max)
        for _ in range(50):
            d = calculate_backoff_delay(attempt_number=5, base_s=1.0, max_backoff_s=8.0)
            assert 0.0 <= d <= 8.0

    def test_retry_after_ms_header(self):
        mock_response = MagicMock()
        mock_response.headers = {"retry-after-ms": "2500"}
        mock_exc = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_response)

        delay = retry_after_seconds(mock_exc)
        assert delay == pytest.approx(2.5, rel=1e-2)

    def test_retry_after_seconds_header(self):
        mock_response = MagicMock()
        mock_response.headers = {"retry-after": "5"}
        mock_exc = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_response)

        delay = retry_after_seconds(mock_exc)
        assert delay == 5.0

    @pytest.mark.asyncio
    async def test_run_with_retries_transient_recovery(self):
        attempts = 0

        async def faulty_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                req = httpx.Request("POST", "https://foundry.azure.com")
                resp = httpx.Response(429, request=req)
                raise openai.RateLimitError("Rate limit", response=resp, body=None)
            return "success_val"

        sleep_calls = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        res = await run_with_retries(faulty_operation, model="gpt-5.6-luna", sleep=fake_sleep)
        assert res == "success_val"
        assert attempts == 3
        assert len(sleep_calls) == 2


class TestAdaptiveConcurrency:
    """Test 3.3 Dynamic Semaphore and Adaptive Rate Stepping."""

    @pytest.mark.asyncio
    async def test_dynamic_semaphore_resize(self):
        sem = DynamicSemaphore(initial_value=2)
        assert sem.capacity == 2

        await sem.acquire()
        await sem.acquire()
        assert sem.value == 0

        # Resize up
        sem.resize(3)
        assert sem.capacity == 3
        assert sem.value == 1

        await sem.acquire()
        assert sem.value == 0

        sem.release()
        sem.release()
        sem.release()
        assert sem.value == 3

    @pytest.mark.asyncio
    async def test_model_limiter_step_down_on_429(self):
        limiter = ModelLimiter(default_limit=8)
        model = "gpt-5.6-luna"

        stats = await limiter.stats(model)
        assert stats.current_capacity == 8

        # Record 429 rate limit
        await limiter.record_feedback(model, status_code=429, is_rate_limited=True)
        stats = await limiter.stats(model)
        assert stats.current_capacity == 4  # 8 // 2

        # Record another 429
        await limiter.record_feedback(model, status_code=429, is_rate_limited=True)
        stats = await limiter.stats(model)
        assert stats.current_capacity == 2  # 4 // 2

    @pytest.mark.asyncio
    async def test_model_limiter_step_up_on_sustained_success(self):
        limiter = ModelLimiter(default_limit=4)
        model = "gpt-5.6-luna"

        # Step down to 2
        await limiter.record_feedback(model, status_code=429, is_rate_limited=True)
        stats = await limiter.stats(model)
        assert stats.current_capacity == 2

        # 5 consecutive successes triggers step-up
        for _ in range(5):
            await limiter.record_feedback(model, status_code=200)

        stats = await limiter.stats(model)
        assert stats.current_capacity == 3


class TestCircuitBreaker:
    """Test 3.3 Circuit Breaker threshold, tripping, and half-open probe."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_after_consecutive_failures(self):
        cb = CircuitBreaker(failure_threshold=5, window_s=30.0, recovery_timeout_s=0.2, name="TestCB")
        assert cb.state == CircuitState.CLOSED

        # 4 failures: should remain closed
        for _ in range(4):
            await cb.record_failure(Exception("simulated 503"))
            assert cb.state == CircuitState.CLOSED

        # 5th failure: should trip to OPEN
        await cb.record_failure(Exception("simulated 503"))
        assert cb.state == CircuitState.OPEN

        # Next check_state should raise CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException) as exc_info:
            await cb.check_state()
        assert "OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        cb = CircuitBreaker(failure_threshold=3, window_s=30.0, recovery_timeout_s=0.1, name="TestCB2")

        for _ in range(3):
            await cb.record_failure(Exception("error"))
        assert cb.state == CircuitState.OPEN

        # Wait recovery timeout
        await asyncio.sleep(0.12)

        # check_state should transition OPEN -> HALF_OPEN
        await cb.check_state()
        assert cb.state == CircuitState.HALF_OPEN

        # A successful trial closes the circuit
        await cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestDualClientAdapter:
    """Test 3.1 & 3.2 Microsoft Foundry Migration & Dual-Client Adapter Pattern."""

    def test_factory_builds_foundry_by_default(self):
        provider = build_llm_provider("foundry")
        assert isinstance(provider, FoundryLLMProvider)
        assert provider.provider_name == "foundry"

    def test_factory_builds_openrouter(self):
        provider = build_llm_provider("openrouter")
        assert isinstance(provider, OpenRouterLLMProvider)
        assert provider.provider_name == "openrouter"

    @pytest.mark.asyncio
    async def test_dual_client_extractor_primary_foundry(self):
        mock_runtime = MagicMock()
        mock_runtime.complete = AsyncMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"ok": true}'))]))

        extractor = DualClientExtractor(provider="foundry", runtime=mock_runtime, fallback_enabled=True)
        assert extractor.active_provider_name == "foundry"

        res = await extractor.complete(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=100,
        )
        assert res.choices[0].message.content == '{"ok": true}'
        mock_runtime.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_dual_client_extractor_failover_to_openrouter(self):
        mock_runtime = MagicMock()
        mock_runtime.complete = AsyncMock(side_effect=RuntimeError("Azure AI Foundry network outage"))

        extractor = DualClientExtractor(provider="foundry", runtime=mock_runtime, fallback_enabled=True)

        # Mock OpenRouter client complete
        mock_or_client = AsyncMock()
        mock_or_client.complete = AsyncMock(return_value="fallback_response")
        extractor._openrouter_client = mock_or_client

        res = await extractor.complete(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=100,
        )
        assert res == "fallback_response"
        mock_or_client.complete.assert_called_once()
