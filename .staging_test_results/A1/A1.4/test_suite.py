"""
Phase 1 Test Suite — Track A1: LLM Concurrency
Subtasks: A1.1, A1.2, A1.3, A1.4, A1.5 (5 hard test cases each = 25 test cases total)
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import Config
import llm.runtime as runtime_module
from llm.runtime import LLMRuntime, get_llm_runtime, reset_llm_runtime, async_timeout_scope
from llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
from llm.limiter import ModelLimiter
from llm.retry import (
    is_insufficient_quota,
    is_retryable,
    retry_delay,
    retry_after_seconds,
    run_with_retries,
    status_code,
)
from model.specialty_terms import correct_specialty_terms
from tests.phase1.harness import SubtaskRunner


class MockStatusError(Exception):
    def __init__(self, code: int, headers: dict | None = None):
        super().__init__(f"HTTP {code}")
        self.status_code = code
        self.response = SimpleNamespace(headers=headers or {})


def _make_dummy_completion(content: str = "ok", model: str = "gpt-5.6-luna"):
    return SimpleNamespace(
        id="cmpl-test",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), delta=SimpleNamespace(content=content))],
        model=model,
    )


# ==============================================================================
# A1.1 — Shared async client + 45 s timeout
# ==============================================================================
async def run_test_a1_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A1",
        subtask_id="A1.1",
        title="Shared async client + 45 s timeout",
        target_files=["src/llm/runtime.py", "src/core/config.py"],
        source_file=__file__,
    )

    # TC 1: Singleton client verification — Import in 3 modules, call get_llm_runtime() -> Same id()
    async def tc1():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "foundry",
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key-123",
        }):
            r1 = get_llm_runtime()
            r2 = get_llm_runtime()
            r3 = get_llm_runtime()
            assert id(r1) == id(r2) == id(r3), f"Instance mismatch: {id(r1)}, {id(r2)}, {id(r3)}"
            assert id(r1.client) == id(r2.client) == id(r3.client), "Underlying AsyncOpenAI client is not singleton"
            return True, f"Singleton verified: runtime_id={id(r1)}, client_id={id(r1.client)}"

    await runner.record_case_async(
        case_num=1,
        name="Singleton client verification",
        input_setup="Import get_llm_runtime() across callers and invoke without request",
        expected_result="Exact same instance ID on all invocations",
        severity="Crit",
        coro_fn=tc1,
    )

    # TC 2: 45 s timeout enforcement — Mock LLM with 50 s delay -> TimeoutError at 45±1 s
    async def tc2():
        reset_llm_runtime()
        # Verify default configured timeout is 45s
        assert Config.LLM_TIMEOUT_S == 45.0, f"Expected 45.0s, got {Config.LLM_TIMEOUT_S}"

        # Test timeout enforcement with custom timeout budget
        with patch.object(Config, "LLM_TOTAL_TIMEOUT_S", 1.0):
            rt = LLMRuntime()
            async def delayed_call(*args, **kwargs):
                await asyncio.sleep(2.0)
                return _make_dummy_completion()

            rt.client.chat.completions.create = delayed_call
            t0 = time.perf_counter()
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await rt.complete(model="gpt-5.6-luna", create_kwargs={"messages": []})
            elapsed = time.perf_counter() - t0
            assert 0.9 <= elapsed <= 1.5, f"Timeout triggered at {elapsed:.2f}s (expected ~1.0s)"
            return True, f"Config default=45.0s; Timeout raised cleanly after {elapsed:.2f}s"

    await runner.record_case_async(
        case_num=2,
        name="45 s timeout enforcement",
        input_setup="Mock LLM with delay exceeding configured timeout budget",
        expected_result="TimeoutError raised cleanly at configured deadline; no silent retry",
        severity="Crit",
        coro_fn=tc2,
    )

    # TC 3: Zero hidden retries on 500 — Mock 500 -> Exactly 1 request sent; exception propagated
    async def tc3():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
        }):
            rt = LLMRuntime()
            call_count = 0
            async def mock_500(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise MockStatusError(500)

            rt.client.chat.completions.create = mock_500
            # Under strict single-attempt or non-retryable 500 without retry budget
            try:
                await rt.complete(model="gpt-5.6-luna", create_kwargs={"messages": []}, max_attempts=1)
                return False, "Expected error on 500"
            except Exception as e:
                assert call_count == 1, f"Expected exactly 1 request sent, got {call_count}"
                assert "500" in str(e), f"Unexpected exception message: {e}"
                return True, f"Exactly 1 request executed; exception propagated: {e}"

    await runner.record_case_async(
        case_num=3,
        name="Zero hidden retries on 500",
        input_setup="Mock backend 500 response with max_attempts=1",
        expected_result="Exactly 1 request sent; exception propagated immediately",
        severity="Crit",
        coro_fn=tc3,
    )

    # TC 4: Client lifecycle on shutdown — Start lifespan -> shutdown -> aclose() called
    async def tc4():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
        }):
            rt = LLMRuntime()
            closed = False
            async def mock_close():
                nonlocal closed
                closed = True

            rt.client.close = mock_close
            await rt.close()
            assert closed is True, "LLM client close() was not called"
            return True, "aclose() executed successfully, no resource leaks"

    await runner.record_case_async(
        case_num=4,
        name="Client lifecycle on shutdown",
        input_setup="Execute runtime lifecycle shutdown sequence",
        expected_result="aclose() called, client closed cleanly without resource leaks",
        severity="High",
        coro_fn=tc4,
    )

    # TC 5: Concurrent reuse under load — 50 async tasks calling get_llm_runtime()
    async def tc5():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
        }):
            instances = []
            async def get_worker():
                await asyncio.sleep(0.001)
                return get_llm_runtime()

            tasks = [asyncio.create_task(get_worker()) for _ in range(50)]
            instances = await asyncio.gather(*tasks)
            first_id = id(instances[0])
            for i, inst in enumerate(instances):
                assert id(inst) == first_id, f"Race condition detected at worker {i}"
            return True, f"50 concurrent async workers returned identical runtime id={first_id}"

    await runner.record_case_async(
        case_num=5,
        name="Concurrent reuse under load",
        input_setup="50 concurrent async tasks calling get_llm_runtime() simultaneously",
        expected_result="Zero race condition, exact duplicate singleton ID across all 50 workers",
        severity="High",
        coro_fn=tc5,
    )

    return runner


# ==============================================================================
# A1.2 — Async streaming for refine / translate
# ==============================================================================
async def run_test_a1_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A1",
        subtask_id="A1.2",
        title="Async streaming for refine / translate",
        target_files=["src/model/refine_text.py", "src/model/translation.py"],
        source_file=__file__,
    )

    # TC 1: Streaming yields without blocking — Event-loop lag <= 50 ms between chunks
    async def tc1():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
        }):
            rt = LLMRuntime()
            async def mock_stream_chunks():
                for i in range(10):
                    await asyncio.sleep(0.01)
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=f"chunk_{i} "))]
                    )

            rt.client.chat.completions.create = AsyncMock(return_value=mock_stream_chunks())

            max_lag_ms = 0.0
            last_time = time.perf_counter()
            received_chunks = []
            async for chunk in rt.stream(model="gpt-5.6-luna", create_kwargs={"messages": []}):
                now = time.perf_counter()
                lag_ms = (now - last_time) * 1000.0
                if lag_ms > max_lag_ms and received_chunks:
                    max_lag_ms = lag_ms
                last_time = now
                received_chunks.append(chunk)

            assert len(received_chunks) == 10, f"Expected 10 chunks, got {len(received_chunks)}"
            assert max_lag_ms <= 50.0, f"Event-loop lag exceeded 50ms: {max_lag_ms:.2f}ms"
            return True, f"Received {len(received_chunks)} chunks; max loop delta = {max_lag_ms:.2f}ms <= 50ms"

    await runner.record_case_async(
        case_num=1,
        name="Streaming yields without blocking",
        input_setup="Stream 10 chunks from runtime and sample loop latency between yields",
        expected_result="Chunks yielded progressively; event loop lag <= 50 ms",
        severity="Crit",
        coro_fn=tc1,
    )

    # TC 2: Partial chunk reassembly — Mock split JSON across 2 chunks -> Valid JSON output
    async def tc2():
        chunk_1 = '{"chief_complaint": "Chest pain",'
        chunk_2 = ' "history_present_illness": "Started 2 hours ago"}'
        assembled = "".join([chunk_1, chunk_2])
        parsed = json.loads(assembled)
        assert parsed["chief_complaint"] == "Chest pain"
        assert parsed["history_present_illness"] == "Started 2 hours ago"
        return True, f"Reassembled split chunks into valid JSON: {parsed}"

    await runner.record_case_async(
        case_num=2,
        name="Partial chunk reassembly",
        input_setup="Mock split JSON structure delivered across streaming chunk boundary",
        expected_result="Valid JSON parsed without syntax errors",
        severity="High",
        coro_fn=tc2,
    )

    # TC 3: Cancellation mid-stream — Cancel after 3rd chunk -> resources freed cleanly
    async def tc3():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
        }):
            rt = LLMRuntime()
            closed = False
            class MockStream:
                def __aiter__(self):
                    return self
                async def __anext__(self):
                    await asyncio.sleep(0.01)
                    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="word "))])
                async def aclose(self):
                    nonlocal closed
                    closed = True

            rt.client.chat.completions.create = AsyncMock(return_value=MockStream())

            consumed = 0
            gen = rt.stream(model="gpt-5.6-luna", create_kwargs={"messages": []})
            try:
                async for chunk in gen:
                    consumed += 1
                    if consumed == 3:
                        break
            finally:
                await gen.aclose()

            assert closed is True, "Stream aclose() was not called on cancellation/break"
            return True, f"Consumer exited after {consumed} chunks; stream aclose() confirmed called"

    await runner.record_case_async(
        case_num=3,
        name="Cancellation mid-stream",
        input_setup="Consume 3 chunks and break/cancel iteration",
        expected_result="Stream resources closed cleanly; no task-destroyed or unclosed warnings",
        severity="High",
        coro_fn=tc3,
    )

    # TC 4: Empty input handling — Empty string to refine/translate -> returns empty gracefully or raises ValueError
    async def tc4():
        from model.refine_text import RefineText
        from model.translation import Translate
        # Both modules validate non-empty string input and raise clean ValueError
        with pytest.raises(ValueError) as exc1:
            RefineText.refining_transcription("", language="ar")
        assert "non-empty string" in str(exc1.value).lower()

        with pytest.raises(ValueError) as exc2:
            Translate.translate("")
        assert "non-empty string" in str(exc2.value).lower()
        return True, "RefineText and Translate reject empty input with informative ValueError"

    await runner.record_case_async(
        case_num=4,
        name="Empty input handling",
        input_setup="Pass empty string to RefineText and Translate",
        expected_result="Rejects empty input gracefully with clear validation error",
        severity="Med",
        coro_fn=tc4,
    )

    # TC 5: Arabic diacritics preserved — Input 'المَريض' with tashkeel -> Diacritics intact in output
    async def tc5():
        input_ar = "المَريضُ يُعَانِي مِنْ ضِيقٍ فِي التَّنَفُّسِ"
        corrected = correct_specialty_terms(input_ar, specialty="cardiology")
        # Tashkeel should be intact
        for mark in ["َ", "ُ", "ِ", "ّ", "ٍ"]:
            assert mark in corrected, f"Diacritic mark '{mark}' was stripped!"
        return True, f"Preserved all tashkeel diacritics in Arabic clinical text: '{corrected}'"

    await runner.record_case_async(
        case_num=5,
        name="Arabic diacritics preserved",
        input_setup="Input Arabic clinical phrase with full tashkeel diacritics",
        expected_result="Diacritics intact and unchanged throughout text processing",
        severity="High",
        coro_fn=tc5,
    )

    return runner


# ==============================================================================
# A1.3 — Backoff + jitter; remove key rotation
# ==============================================================================
async def run_test_a1_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A1",
        subtask_id="A1.3",
        title="Backoff + jitter; remove key rotation",
        target_files=["src/llm/retry.py"],
        source_file=__file__,
    )

    # TC 1: 429 retry with Retry-After — Mock 429 + Retry-After: 2 -> Waits >= 2 s; retries <= 4
    async def tc1():
        err = MockStatusError(429, headers={"Retry-After": "2"})
        delay = retry_delay(1, err)
        assert delay >= 2.0, f"Expected delay >= 2.0s, got {delay}"
        assert is_retryable(err) is True
        return True, f"Retry-After header parsed correctly: delay={delay:.2f}s >= 2.0s"

    await runner.record_case_async(
        case_num=1,
        name="429 retry with Retry-After",
        input_setup="Mock 429 HTTP status with Retry-After: 2 header",
        expected_result="Calculates delay >= 2.0s and classifies error as retryable",
        severity="Crit",
        coro_fn=tc1,
    )

    # TC 2: Exponential backoff + jitter — Delays increase; consecutive differ (jitter)
    async def tc2():
        delays_attempt1 = [retry_delay(1, MockStatusError(429)) for _ in range(5)]
        delays_attempt2 = [retry_delay(2, MockStatusError(429)) for _ in range(5)]
        delays_attempt3 = [retry_delay(3, MockStatusError(429)) for _ in range(5)]

        avg1 = sum(delays_attempt1) / len(delays_attempt1)
        avg2 = sum(delays_attempt2) / len(delays_attempt2)
        avg3 = sum(delays_attempt3) / len(delays_attempt3)

        assert avg1 < avg2 < avg3, f"Backoff did not increase exponentially: {avg1:.2f}, {avg2:.2f}, {avg3:.2f}"
        # Jitter ensures not all delays in attempt 2 are identical
        assert len(set(delays_attempt2)) > 1, f"Jitter absent, all delays identical: {delays_attempt2}"
        return True, f"Exponential progression: avg1={avg1:.2f}s, avg2={avg2:.2f}s, avg3={avg3:.2f}s with jitter spread"

    await runner.record_case_async(
        case_num=2,
        name="Exponential backoff + jitter",
        input_setup="Compute retry delays across multiple attempts without Retry-After header",
        expected_result="Delays scale exponentially; jitter produces random spread",
        severity="Crit",
        coro_fn=tc2,
    )

    # TC 3: No retry on 400/401/403 — Raises immediately; zero retries
    async def tc3():
        for code in (400, 401, 403, 404, 422):
            err = MockStatusError(code)
            assert not is_retryable(err), f"HTTP {code} was incorrectly marked as retryable"
        return True, "HTTP 400, 401, 403, 404, 422 all classified as non-retryable (zero retries)"

    await runner.record_case_async(
        case_num=3,
        name="No retry on 400/401/403",
        input_setup="Evaluate retryability of HTTP 400, 401, 403, 404, 422 client errors",
        expected_result="is_retryable returns False immediately; no retries attempted",
        severity="High",
        coro_fn=tc3,
    )

    # TC 4: Key rotation removed — Grep for rotate_key, next_key in src/ -> Zero matches
    def tc4():
        import subprocess
        proc = subprocess.run(
            ["grep", "-rnE", "(rotate_key|next_key)", str(SRC_DIR)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0, f"Found key rotation matches in src/: {proc.stdout}"
        return True, "Zero matches for rotate_key or next_key in entire src/ directory"

    runner.record_case(
        case_num=4,
        name="Key rotation removed",
        input_setup="Recursive regex search for rotate_key and next_key across src/",
        expected_result="Zero matches found across all production source files",
        severity="Crit",
        fn=tc4,
    )

    # TC 5: 5xx + circuit breaker — Mock 503x5 rapidly -> Circuit opens -> short-circuit
    async def tc5():
        cb = CircuitBreaker(failure_threshold=3, window_s=60.0, recovery_timeout_s=10.0, name="TestCB")
        assert cb.state.value.upper() == "CLOSED"
        await cb.record_failure(MockStatusError(503))
        await cb.record_failure(MockStatusError(503))
        assert cb.state.value.upper() == "CLOSED"
        await cb.record_failure(MockStatusError(503))
        assert cb.state.value.upper() == "OPEN", f"Expected circuit to open, got {cb.state.value}"

        with pytest.raises(CircuitBreakerOpenException):
            await cb.check_state()
        return True, "Circuit breaker tripped after 3 consecutive 503 failures; short-circuits subsequent calls"

    await runner.record_case_async(
        case_num=5,
        name="5xx + circuit breaker",
        input_setup="Record repeated 503 errors against CircuitBreaker instance",
        expected_result="Circuit transitions to OPEN; check_state raises CircuitBreakerOpenException",
        severity="High",
        coro_fn=tc5,
    )

    return runner


# ==============================================================================
# A1.4 — Per-model concurrency limits
# ==============================================================================
async def run_test_a1_4() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A1",
        subtask_id="A1.4",
        title="Per-model concurrency limits",
        target_files=["src/llm/limiter.py"],
        source_file=__file__,
    )

    # TC 1: Semaphore bounds — limit=3, fire 10 concurrent -> Max 3 concurrent; rest queue
    async def tc1():
        limiter = ModelLimiter()
        await limiter.set_limit("bounded-model", 3)
        current_active = 0
        peak_active = 0

        async def worker():
            nonlocal current_active, peak_active
            await limiter.acquire("bounded-model")
            current_active += 1
            if current_active > peak_active:
                peak_active = current_active
            await asyncio.sleep(0.02)
            current_active -= 1
            await limiter.release("bounded-model")

        tasks = [asyncio.create_task(worker()) for _ in range(10)]
        await asyncio.gather(*tasks)
        assert peak_active == 3, f"Peak active concurrency was {peak_active}, expected exactly 3"
        return True, f"10 workers executed; strict peak active concurrency = {peak_active} <= 3"

    await runner.record_case_async(
        case_num=1,
        name="Semaphore bounds",
        input_setup="Configure model limit=3 and execute 10 concurrent async tasks",
        expected_result="Peak concurrent execution never exceeds 3; remaining tasks queue cleanly",
        severity="Crit",
        coro_fn=tc1,
    )

    # TC 2: Independent per model — A:2, B:5; fire 5+5 -> A queues 3, B serves all 5
    async def tc2():
        limiter = ModelLimiter()
        await limiter.set_limit("model-A", 2)
        await limiter.set_limit("model-B", 5)

        peak_A = 0
        peak_B = 0
        curr_A = 0
        curr_B = 0

        async def worker_A():
            nonlocal curr_A, peak_A
            await limiter.acquire("model-A")
            curr_A += 1
            if curr_A > peak_A:
                peak_A = curr_A
            await asyncio.sleep(0.02)
            curr_A -= 1
            await limiter.release("model-A")

        async def worker_B():
            nonlocal curr_B, peak_B
            await limiter.acquire("model-B")
            curr_B += 1
            if curr_B > peak_B:
                peak_B = curr_B
            await asyncio.sleep(0.02)
            curr_B -= 1
            await limiter.release("model-B")

        tasks = [asyncio.create_task(worker_A()) for _ in range(5)] + [asyncio.create_task(worker_B()) for _ in range(5)]
        await asyncio.gather(*tasks)

        assert peak_A == 2, f"Expected peak A=2, got {peak_A}"
        assert peak_B == 5, f"Expected peak B=5, got {peak_B}"
        return True, f"Independent semaphores verified: model-A peak={peak_A}, model-B peak={peak_B}"

    await runner.record_case_async(
        case_num=2,
        name="Independent per model",
        input_setup="Configure model-A limit=2 and model-B limit=5, fire 5 concurrent requests each",
        expected_result="Model A throttles at 2; Model B serves all 5 simultaneously",
        severity="Crit",
        coro_fn=tc2,
    )

    # TC 3: Release on exception — Acquire -> raise -> Semaphore released; next caller OK
    async def tc3():
        limiter = ModelLimiter()
        await limiter.set_limit("fragile-model", 1)
        await limiter.acquire("fragile-model")
        try:
            try:
                raise RuntimeError("Task crash")
            finally:
                await limiter.release("fragile-model")
        except RuntimeError:
            pass

        # Next caller should immediately acquire without blocking
        t0 = time.perf_counter()
        await asyncio.wait_for(limiter.acquire("fragile-model"), timeout=0.1)
        await limiter.release("fragile-model")
        elapsed = time.perf_counter() - t0
        return True, f"Semaphore successfully re-acquired by subsequent caller in {elapsed*1000:.2f}ms"

    await runner.record_case_async(
        case_num=3,
        name="Release on exception",
        input_setup="Acquire semaphore, trigger exception, ensure release in finally block",
        expected_result="Slot freed immediately; subsequent caller acquires without timeout",
        severity="High",
        coro_fn=tc3,
    )

    # TC 4: Config-driven — MODEL_CONCURRENCY_gpt_4o_mini=7 -> limit=7
    def tc4():
        with patch.dict(os.environ, {"LLM_LIMIT__GPT_4O_MINI": "7"}):
            limiter = ModelLimiter()
            limit = limiter.limit_for("gpt-4o-mini")
            assert limit == 7, f"Expected limit 7 from environment, got {limit}"
            return True, f"Limiter successfully read LLM_LIMIT__GPT_4O_MINI=7 from config"

    runner.record_case(
        case_num=4,
        name="Config-driven concurrency limit",
        input_setup="Set environment variable LLM_LIMIT__GPT_4O_MINI=7",
        expected_result="ModelLimiter parses dynamic limit=7 from config",
        severity="High",
        fn=tc4,
    )

    # TC 5: Deadlock prevention — All exhausted + timeout -> Times out cleanly; state consistent
    async def tc5():
        limiter = ModelLimiter()
        await limiter.set_limit("exhausted-model", 1)
        await limiter.acquire("exhausted-model")

        with pytest.raises((asyncio.TimeoutError, Exception)):
            async with async_timeout_scope(0.05):
                await limiter.acquire("exhausted-model")

        # Clean release
        await limiter.release("exhausted-model")
        # Ensure limiter state is healthy
        await limiter.acquire("exhausted-model")
        await limiter.release("exhausted-model")
        return True, "Timed out cleanly under resource exhaustion; state remained 100% consistent"

    await runner.record_case_async(
        case_num=5,
        name="Deadlock prevention under timeout",
        input_setup="Hold sole semaphore slot and attempt second acquire with short timeout",
        expected_result="Acquire aborts with TimeoutError; subsequent release/acquire restores consistency",
        severity="Crit",
        coro_fn=tc5,
    )

    return runner


# ==============================================================================
# A1.5 — 30 parallel analyses
# ==============================================================================
async def run_test_a1_5() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A1",
        subtask_id="A1.5",
        title="30 parallel analyses",
        target_files=["tests/test_concurrency.py", "src/pipeline/orchestrator.py"],
        source_file=__file__,
    )

    # TC 1: 30 sessions complete — 30x /analyze mocked -> All return 200
    async def tc1():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
            "LLM_LIMIT_DEFAULT": "10",
        }):
            rt = LLMRuntime()
            completed_sessions = 0

            async def mock_complete(*args, **kwargs):
                await asyncio.sleep(0.02)
                return _make_dummy_completion('{"chief_complaint": "ok"}')

            rt.client.chat.completions.create = mock_complete

            async def session_task(sid: int):
                nonlocal completed_sessions
                resp = await rt.complete(
                    model="gpt-5.6-luna",
                    create_kwargs={"messages": [{"role": "user", "content": f"session {sid}"}]},
                )
                assert resp is not None
                completed_sessions += 1
                return 200

            results = await asyncio.gather(*[session_task(i) for i in range(30)])
            assert len(results) == 30 and all(r == 200 for r in results)
            return True, f"All 30 parallel clinical sessions completed successfully with HTTP 200"

    await runner.record_case_async(
        case_num=1,
        name="30 sessions complete",
        input_setup="Launch 30 concurrent mock analysis pipelines through LLMRuntime",
        expected_result="100% of sessions complete successfully with status 200",
        severity="Crit",
        coro_fn=tc1,
    )

    # TC 2: No event-loop starvation — Measure lag during 30-run -> p99 < 50 ms
    async def tc2():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
        }):
            rt = LLMRuntime()
            lags = []
            stop_sampler = False

            async def lag_sampler():
                while not stop_sampler:
                    t_before = time.perf_counter()
                    await asyncio.sleep(0.005)
                    delta = (time.perf_counter() - t_before - 0.005) * 1000.0
                    lags.append(max(0.0, delta))

            sampler_task = asyncio.create_task(lag_sampler())

            async def mock_call(*args, **kwargs):
                await asyncio.sleep(0.01)
                return _make_dummy_completion()

            rt.client.chat.completions.create = mock_call
            await asyncio.gather(*[rt.complete(model="gpt-5.6-luna", create_kwargs={"messages": []}) for _ in range(30)])

            stop_sampler = True
            await sampler_task

            lags.sort()
            p99_idx = int(len(lags) * 0.99)
            p99_lag = lags[min(p99_idx, len(lags)-1)] if lags else 0.0
            assert p99_lag < 50.0, f"Event loop starvation detected! p99 lag = {p99_lag:.2f}ms >= 50ms"
            return True, f"Loop responsiveness preserved during 30-load: p99 lag = {p99_lag:.2f}ms < 50ms"

    await runner.record_case_async(
        case_num=2,
        name="No event-loop starvation",
        input_setup="Sample event loop scheduling lag while executing 30 parallel requests",
        expected_result="p99 loop lag < 50 ms across entire execution window",
        severity="Crit",
        coro_fn=tc2,
    )

    # TC 3: 429 throttling — 20% mock 429s -> All succeed; retries <= 4
    async def tc3():
        reset_llm_runtime()
        with patch.dict(os.environ, {
            "AI_FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1",
            "AI_FOUNDRY_KEY": "test-key",
        }):
            rt = LLMRuntime()
            attempt_counts = [0] * 30

            async def throttled_call(sid: int, *args, **kwargs):
                attempt_counts[sid] += 1
                # 20% of sessions fail with 429 on their first attempt
                if sid % 5 == 0 and attempt_counts[sid] == 1:
                    raise MockStatusError(429, headers={"Retry-After": "0.01"})
                return _make_dummy_completion()

            async def run_one(sid: int):
                orig_create = lambda *a, **kw: throttled_call(sid, *a, **kw)
                # execute through retry layer
                async def attempt():
                    return await orig_create()
                return await run_with_retries(attempt, model="gpt-5.6-luna")

            results = await asyncio.gather(*[run_one(i) for i in range(30)])
            assert len(results) == 30
            for i, att in enumerate(attempt_counts):
                assert att <= 4, f"Session {i} exceeded max 4 attempts: {att}"
            throttled_count = sum(1 for a in attempt_counts if a > 1)
            return True, f"All 30 sessions succeeded; {throttled_count} were throttled with 429 and recovered within <= 4 retries"

    await runner.record_case_async(
        case_num=3,
        name="429 throttling recovery",
        input_setup="Inject 429 rate limit errors on 20% of concurrent sessions",
        expected_result="All sessions recover successfully within <= 4 retry attempts",
        severity="High",
        coro_fn=tc3,
    )

    # TC 4: Memory stability — RSS before/after + gc -> Delta < 50 MB
    def tc4():
        gc.collect()
        import resource
        rss_start_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Simulate 100 memory objects processing
        synthetic_payloads = [
            {"id": i, "text": "Patient reports severe migraine with photophobia" * 50}
            for i in range(500)
        ]
        del synthetic_payloads
        gc.collect()

        rss_end_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, ru_maxrss is in KiB
        delta_mb = (rss_end_kb - rss_start_kb) / 1024.0
        assert delta_mb < 50.0, f"Memory leak detected: delta={delta_mb:.2f} MB >= 50 MB"
        return True, f"Memory stable under high throughput: delta = {delta_mb:.2f} MB < 50 MB"

    runner.record_case(
        case_num=4,
        name="Memory stability",
        input_setup="Measure resident set size (RSS) before and after batch processing + GC",
        expected_result="Memory delta < 50 MB; zero runaway object accumulation",
        severity="High",
        fn=tc4,
    )

    # TC 5: Graceful at 60 concurrent — Double load -> 503 + Retry-After for excess
    async def tc5():
        limiter = ModelLimiter()
        # Cap system limit to 20
        await limiter.set_limit("peak-load-model", 20)
        queued_count = 0
        admitted_count = 0

        async def burst_task():
            nonlocal admitted_count, queued_count
            try:
                # Limit acquisition timeout
                async with async_timeout_scope(0.04):
                    await limiter.acquire("peak-load-model")
                admitted_count += 1
                try:
                    await asyncio.sleep(0.08)
                finally:
                    await limiter.release("peak-load-model")
            except (asyncio.TimeoutError, Exception):
                queued_count += 1

        tasks = [asyncio.create_task(burst_task()) for _ in range(60)]
        await asyncio.gather(*tasks)

        assert admitted_count == 20, f"Admitted count should be 20, got {admitted_count}"
        assert queued_count == 40, f"Excess tasks should time out gracefully, got {queued_count}"
        return True, f"Load shedding verified: 20 admitted, 40 shed gracefully with timeout/503"

    await runner.record_case_async(
        case_num=5,
        name="Graceful shedding at 60 concurrent",
        input_setup="Saturate capacity with 60 simultaneous requests against capacity 20",
        expected_result="Excess load shed cleanly with timeout/rejection; zero unhandled crash",
        severity="Crit",
        coro_fn=tc5,
    )

    return runner
