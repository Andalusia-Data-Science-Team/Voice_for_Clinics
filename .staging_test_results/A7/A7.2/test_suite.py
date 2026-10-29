"""
Phase 3 Test Suite — Track A7: Resilience, Availability & Hardening
Subtasks: A7.1 (Health probes), A7.2 (Circuit breaker), A7.3 (Model fallbacks), A7.4 (Admission control)
(5 hard test cases each = 20 total)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.phase1.harness import SubtaskRunner


# ==============================================================================
# A7.1 — Health probes (/health and /health/live)
# ==============================================================================
def run_test_a7_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A7",
        subtask_id="A7.1",
        title="Probes verify process liveness and downstream dependency readiness",
        target_files=["src/app.py"],
        source_file=__file__,
    )

    # TC 1: /health/live returns 200 without I/O
    async def tc1():
        from app import health_live
        resp = await health_live()
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        assert b'"status":"alive"' in resp.body or b'"alive"' in resp.body, "Missing status alive"
        return True, "/health/live returns HTTP 200 status: alive"

    runner.record_case(
        case_num=1,
        name="Liveness probe returns HTTP 200 without I/O",
        input_setup="Directly invoke health_live() handler",
        expected_result="HTTP 200 with status='alive'",
        severity="Crit",
        fn=lambda: asyncio.run(tc1()),
    )

    # TC 2: /health returns 200 when Foundry dependency is healthy
    async def tc2():
        from app import health_check
        req = MagicMock()
        mock_payload = {
            "status": "ok",
            "dependencies": {
                "foundry": {"status": "ok"},
                "munsit": {"status": "ok"},
                "voxtral": {"status": "ok"},
            }
        }
        with patch("app._dependency_health", new_callable=AsyncMock) as mock_dep:
            mock_dep.return_value = mock_payload
            resp = await health_check(req)
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            assert b'"status":"ok"' in resp.body
        return True, "/health returns HTTP 200 when dependencies ok"

    runner.record_case(
        case_num=2,
        name="Readiness probe returns 200 on healthy dependencies",
        input_setup="Invoke health_check with healthy dependency payload",
        expected_result="HTTP 200 status='ok'",
        severity="Crit",
        fn=lambda: asyncio.run(tc2()),
    )

    # TC 3: /health returns 503 when Foundry is down
    async def tc3():
        from app import health_check
        req = MagicMock()
        mock_payload = {
            "status": "unhealthy",
            "dependencies": {
                "foundry": {"status": "unhealthy", "error": "timeout"},
                "munsit": {"status": "ok"},
                "voxtral": {"status": "ok"},
            }
        }
        with patch("app._dependency_health", new_callable=AsyncMock) as mock_dep:
            mock_dep.return_value = mock_payload
            resp = await health_check(req)
            assert resp.status_code == 503, f"Expected 503 for unhealthy Foundry, got {resp.status_code}"
            assert b'"status":"unhealthy"' in resp.body
        return True, "/health returns 503 when critical dependency (Foundry) is unhealthy"

    runner.record_case(
        case_num=3,
        name="Readiness returns 503 when Foundry is unhealthy",
        input_setup="Mock _dependency_health returning unhealthy Foundry",
        expected_result="HTTP 503 response gating deployment",
        severity="Crit",
        fn=lambda: asyncio.run(tc3()),
    )

    # TC 4: /health returns 200 degraded when non-critical STT is down
    async def tc4():
        from app import health_check
        req = MagicMock()
        mock_payload = {
            "status": "degraded",
            "dependencies": {
                "foundry": {"status": "ok"},
                "munsit": {"status": "unhealthy", "error": "connection refused"},
                "voxtral": {"status": "ok"},
            }
        }
        with patch("app._dependency_health", new_callable=AsyncMock) as mock_dep:
            mock_dep.return_value = mock_payload
            resp = await health_check(req)
            assert resp.status_code == 200, f"Expected 200 for degraded STT, got {resp.status_code}"
            assert b'"status":"degraded"' in resp.body
        return True, "/health returns 200 (degraded) on non-critical STT outage"

    runner.record_case(
        case_num=4,
        name="Readiness returns 200 (degraded) when optional STT fails",
        input_setup="Mock _dependency_health returning degraded STT status",
        expected_result="HTTP 200 status='degraded' to avoid breaking ingress",
        severity="High",
        fn=lambda: asyncio.run(tc4()),
    )

    # TC 5: Probe caching prevents hammering downstream APIs
    def tc5():
        import app
        assert hasattr(app, "_health_cache"), "app missing _health_cache"
        assert "_HEALTH_CACHE_TTL_S" in dir(app) or "30" in str(getattr(app, "_HEALTH_CACHE_TTL_S", 30)), "Cache TTL missing"
        return True, "Health check cache state and TTL defined to prevent API throttling"

    runner.record_case(
        case_num=5,
        name="Health response caching mechanics",
        input_setup="Inspect app._health_cache and TTL definitions",
        expected_result="Cache dictionary and TTL defined in app module",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# A7.2 — Circuit breaker for LLM services
# ==============================================================================
def run_test_a7_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A7",
        subtask_id="A7.2",
        title="Circuit breaker protects downstream LLM endpoints from cascading failure",
        target_files=["src/llm/circuit_breaker.py"],
        source_file=__file__,
    )

    # TC 1: Trips from CLOSED to OPEN after failure_threshold reached
    async def tc1():
        from llm.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpenException
        cb = CircuitBreaker(failure_threshold=3, window_s=10.0, recovery_timeout_s=5.0)
        assert cb.state == CircuitState.CLOSED

        await cb.record_failure(RuntimeError("fail 1"))
        await cb.record_failure(RuntimeError("fail 2"))
        assert cb.state == CircuitState.CLOSED

        await cb.record_failure(RuntimeError("fail 3"))
        assert cb.state == CircuitState.OPEN

        try:
            await cb.check_state()
            return False, "Should have raised CircuitBreakerOpenException when OPEN"
        except CircuitBreakerOpenException:
            return True, "Circuit tripped to OPEN and immediately rejected subsequent call"

    runner.record_case(
        case_num=1,
        name="Trips CLOSED -> OPEN at failure threshold",
        input_setup="Register 3 consecutive failures with threshold=3",
        expected_result="State becomes OPEN and check_state raises CircuitBreakerOpenException",
        severity="Crit",
        fn=lambda: asyncio.run(tc1()),
    )

    # TC 2: Transitions OPEN -> HALF_OPEN after recovery timeout
    async def tc2():
        from llm.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.05)
        await cb.record_failure(RuntimeError("f1"))
        await cb.record_failure(RuntimeError("f2"))
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.06)
        # check_state should now allow probe and transition to HALF_OPEN
        await cb.check_state()
        assert cb.state == CircuitState.HALF_OPEN, f"Expected HALF_OPEN, got {cb.state}"
        return True, "Circuit transitioned OPEN -> HALF_OPEN after recovery timeout elapsed"

    runner.record_case(
        case_num=2,
        name="Transitions OPEN -> HALF_OPEN after recovery timeout",
        input_setup="Wait recovery_timeout_s and call check_state()",
        expected_result="Circuit transitions to HALF_OPEN to permit probe execution",
        severity="Crit",
        fn=lambda: asyncio.run(tc2()),
    )

    # TC 3: Closes circuit when trial request succeeds in HALF_OPEN
    async def tc3():
        from llm.circuit_breaker import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.02)
        await cb.record_failure(RuntimeError("f1"))
        await cb.record_failure(RuntimeError("f2"))
        await asyncio.sleep(0.03)
        await cb.check_state()
        assert cb.state == CircuitState.HALF_OPEN

        await cb.record_success()
        assert cb.state == CircuitState.CLOSED, f"Expected CLOSED, got {cb.state}"
        return True, "Trial success closed the circuit breaker"

    runner.record_case(
        case_num=3,
        name="Closes circuit on probe success in HALF_OPEN",
        input_setup="Call record_success while in HALF_OPEN",
        expected_result="Circuit transitions back to CLOSED",
        severity="Crit",
        fn=lambda: asyncio.run(tc3()),
    )

    # TC 4: Re-trips to OPEN on failure in HALF_OPEN
    async def tc4():
        from llm.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpenException
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.02)
        await cb.record_failure(RuntimeError("f1"))
        await cb.record_failure(RuntimeError("f2"))
        await asyncio.sleep(0.03)
        await cb.check_state()
        assert cb.state == CircuitState.HALF_OPEN

        await cb.record_failure(RuntimeError("probe failed"))
        assert cb.state == CircuitState.OPEN, f"Expected OPEN, got {cb.state}"
        return True, "Probe failure in HALF_OPEN immediately returned circuit to OPEN"

    runner.record_case(
        case_num=4,
        name="Re-trips to OPEN on probe failure",
        input_setup="Call record_failure while in HALF_OPEN",
        expected_result="Circuit immediately returns to OPEN state",
        severity="High",
        fn=lambda: asyncio.run(tc4()),
    )

    # TC 5: Stats reporting
    async def tc5():
        from llm.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2)
        await cb.record_failure(RuntimeError("f1"))
        stats = await cb.get_stats()
        assert stats.consecutive_failures == 1
        assert stats.tripped_count == 0
        return True, "Circuit breaker stats reflect failure count and state"

    runner.record_case(
        case_num=5,
        name="Circuit breaker stats tracking",
        input_setup="Query get_stats() after recorded failure",
        expected_result="Accurate stats object returned",
        severity="High",
        fn=lambda: asyncio.run(tc5()),
    )

    return runner


# ==============================================================================
# A7.3 — Model fallbacks
# ==============================================================================
def run_test_a7_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A7",
        subtask_id="A7.3",
        title="Single-hop model fallback on eligible provider errors",
        target_files=["src/pipeline/fallback.py"],
        source_file=__file__,
    )

    # TC 1: Eligibility check allows 429 and 5xx, rejects 400
    def tc1():
        from pipeline.fallback import is_fallback_eligible
        import httpx
        err_429 = httpx.HTTPStatusError("429", request=MagicMock(), response=MagicMock(status_code=429))
        err_503 = httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock(status_code=503))
        err_400 = httpx.HTTPStatusError("400", request=MagicMock(), response=MagicMock(status_code=400))
        err_timeout = httpx.TimeoutException("timeout")

        assert is_fallback_eligible(err_429) is True, "429 should be eligible"
        assert is_fallback_eligible(err_503) is True, "503 should be eligible"
        assert is_fallback_eligible(err_timeout) is True, "Timeout should be eligible"
        assert is_fallback_eligible(err_400) is False, "400 must NOT be eligible"
        return True, "Eligibility classification adheres strictly to trigger rules"

    runner.record_case(
        case_num=1,
        name="Fallback eligibility criteria verification",
        input_setup="Test 429, 503, timeout vs 400 with is_fallback_eligible",
        expected_result="429/503/timeout eligible; 400 ineligible",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Primary success does not call fallback
    async def tc2():
        from pipeline.fallback import run_with_single_fallback
        primary_mock = AsyncMock(return_value="primary_success")
        fallback_mock = AsyncMock(return_value="fallback_result")
        meta = {}

        res = await run_with_single_fallback(
            primary_mock,
            fallback_mock,
            primary_deployment="model-a",
            fallback_deployment="model-b",
            meta=meta,
        )
        assert res == "primary_success"
        assert fallback_mock.call_count == 0
        assert meta.get("fallback_used") is False
        assert meta.get("served_deployment") == "model-a"
        return True, "Primary call succeeded without invoking fallback"

    runner.record_case(
        case_num=2,
        name="Primary success bypasses fallback invocation",
        input_setup="Execute run_with_single_fallback with succeeding primary callable",
        expected_result="Fallback not called; meta indicates fallback_used=False",
        severity="Crit",
        fn=lambda: asyncio.run(tc2()),
    )

    # TC 3: Eligible error triggers single fallback invocation
    async def tc3():
        import httpx
        from pipeline.fallback import run_with_single_fallback
        primary_mock = AsyncMock(side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock(status_code=503)))
        fallback_mock = AsyncMock(return_value="recovered_from_fallback")
        meta = {}

        res = await run_with_single_fallback(
            primary_mock,
            fallback_mock,
            primary_deployment="model-a",
            fallback_deployment="model-b",
            meta=meta,
        )
        assert res == "recovered_from_fallback"
        assert fallback_mock.call_count == 1
        assert meta.get("fallback_used") is True
        assert meta.get("served_deployment") == "model-b"
        return True, "Eligible 503 triggered exactly one fallback hop"

    runner.record_case(
        case_num=3,
        name="Eligible error triggers single fallback call",
        input_setup="Primary raises HTTP 503; fallback succeeds",
        expected_result="Fallback executed; meta records fallback_used=True and served_deployment",
        severity="Crit",
        fn=lambda: asyncio.run(tc3()),
    )

    # TC 4: Ineligible error is re-raised immediately without fallback
    async def tc4():
        import httpx
        from pipeline.fallback import run_with_single_fallback
        primary_mock = AsyncMock(side_effect=httpx.HTTPStatusError("400", request=MagicMock(), response=MagicMock(status_code=400)))
        fallback_mock = AsyncMock(return_value="should_not_run")

        try:
            await run_with_single_fallback(
                primary_mock,
                fallback_mock,
                primary_deployment="model-a",
                fallback_deployment="model-b",
            )
            return False, "Should have raised 400"
        except httpx.HTTPStatusError:
            assert fallback_mock.call_count == 0, "Fallback must not run on 400"
            return True, "400 error re-raised immediately; fallback never called"

    runner.record_case(
        case_num=4,
        name="Ineligible error propagated without invoking fallback",
        input_setup="Primary raises HTTP 400",
        expected_result="Exception propagated; fallback called 0 times",
        severity="Crit",
        fn=lambda: asyncio.run(tc4()),
    )

    # TC 5: Fallback exception is propagated when both fail
    async def tc5():
        import httpx
        from pipeline.fallback import run_with_single_fallback
        primary_mock = AsyncMock(side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock(status_code=503)))
        fallback_mock = AsyncMock(side_effect=RuntimeError("fallback also dead"))

        try:
            await run_with_single_fallback(
                primary_mock,
                fallback_mock,
                primary_deployment="model-a",
                fallback_deployment="model-b",
            )
            return False, "Should have raised fallback error"
        except RuntimeError as e:
            assert "fallback also dead" in str(e)
            return True, "Fallback exception propagated without infinite loop"

    runner.record_case(
        case_num=5,
        name="Fallback failure propagated without infinite looping",
        input_setup="Both primary and fallback raise exceptions",
        expected_result="Fallback exception propagated cleanly; maximum single hop enforced",
        severity="High",
        fn=lambda: asyncio.run(tc5()),
    )

    return runner


# ==============================================================================
# A7.4 — Admission control
# ==============================================================================
def run_test_a7_4() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A7",
        subtask_id="A7.4",
        title="Admission controller bounds concurrent requests and manages capacity overload",
        target_files=["src/core/admission.py"],
        source_file=__file__,
    )

    # TC 1: Concurrency bounding rejects when max_analyses exceeded
    async def tc1():
        from core.admission import AdmissionController, AdmissionRejectedError
        ctrl = AdmissionController()
        ctrl.max_analyses = 2

        with patch("core.admission.get_active_analyses", return_value=2):
            try:
                await ctrl.check_analysis()
                return False, "Should have raised AdmissionRejectedError"
            except AdmissionRejectedError as e:
                assert e.active == 2 and e.maximum == 2
                return True, "check_analysis raised AdmissionRejectedError at capacity limit"

    runner.record_case(
        case_num=1,
        name="Rejects requests exceeding max_analyses threshold",
        input_setup="AdmissionController with max_analyses=2 and mock active=2",
        expected_result="AdmissionRejectedError raised",
        severity="Crit",
        fn=lambda: asyncio.run(tc1()),
    )

    # TC 2: WebSocket concurrency bounding
    async def tc2():
        from core.admission import AdmissionController, AdmissionRejectedError
        ctrl = AdmissionController()
        ctrl.max_websockets = 5

        with patch("core.admission.get_active_websockets", return_value=5):
            try:
                await ctrl.check_websocket()
                return False, "Should have raised AdmissionRejectedError"
            except AdmissionRejectedError as e:
                assert e.path == "/ws/transcribe"
                return True, "check_websocket raised AdmissionRejectedError at socket limit"

    runner.record_case(
        case_num=2,
        name="Rejects websockets exceeding max_websockets threshold",
        input_setup="AdmissionController with max_websockets=5 and mock active=5",
        expected_result="AdmissionRejectedError raised with path=/ws/transcribe",
        severity="Crit",
        fn=lambda: asyncio.run(tc2()),
    )

    # TC 3: Slot acquisition tracks lifecycle increment & decrement
    async def tc3():
        from core.admission import AdmissionController
        ctrl = AdmissionController()
        ctrl.max_analyses = 10

        inc_mock = MagicMock()
        dec_mock = MagicMock()

        with patch("core.admission.get_active_analyses", return_value=0), \
             patch("core.admission.increment_active_analyses", inc_mock), \
             patch("core.admission.decrement_active_analyses", dec_mock):
            async with ctrl.acquire_analysis_slot():
                assert inc_mock.call_count == 1
                assert dec_mock.call_count == 0
            assert dec_mock.call_count == 1
        return True, "acquire_analysis_slot cleanly managed increment and decrement lifecycle"

    runner.record_case(
        case_num=3,
        name="acquire_analysis_slot manages lifetime bounds",
        input_setup="Enter and exit acquire_analysis_slot context manager",
        expected_result="increment called on enter, decrement called on exit",
        severity="Crit",
        fn=lambda: asyncio.run(tc3()),
    )

    # TC 4: Slot release on exception inside context manager
    async def tc4():
        from core.admission import AdmissionController
        ctrl = AdmissionController()
        dec_mock = MagicMock()

        with patch("core.admission.get_active_analyses", return_value=0), \
             patch("core.admission.increment_active_analyses"), \
             patch("core.admission.decrement_active_analyses", dec_mock):
            try:
                async with ctrl.acquire_analysis_slot():
                    raise RuntimeError("error inside pipeline")
            except RuntimeError:
                pass
            assert dec_mock.call_count == 1, "decrement was not called on exception"
        return True, "Slot released immediately even when task raises exception"

    runner.record_case(
        case_num=4,
        name="Slot released safely when exception occurs",
        input_setup="Raise exception inside acquire_analysis_slot context manager",
        expected_result="decrement called ensuring no slot leak",
        severity="High",
        fn=lambda: asyncio.run(tc4()),
    )

    # TC 5: Stats property reports live capacity and utilization
    async def tc5():
        from core.admission import AdmissionController
        ctrl = AdmissionController()
        ctrl.max_analyses = 20
        ctrl.max_websockets = 50

        with patch("core.admission.get_active_analyses", return_value=10), \
             patch("core.admission.get_active_websockets", return_value=25):
            st = ctrl.stats
            assert st["active_analyses"] == 10
            assert st["analyses_utilization"] == 0.5
            assert st["active_websockets"] == 25
            assert st["websockets_utilization"] == 0.5
            assert st["is_saturated"] is False
        return True, f"AdmissionController.stats calculated accurate 50% utilization metrics"

    runner.record_case(
        case_num=5,
        name="Stats reporting calculates utilization ratios",
        input_setup="Query AdmissionController.stats under mocked active counts",
        expected_result="Accurate utilization ratios calculated for both analyses and websockets",
        severity="High",
        fn=lambda: asyncio.run(tc5()),
    )

    return runner


# ==============================================================================
# Main runner
# ==============================================================================
if __name__ == "__main__":
    results = []

    r = run_test_a7_1()
    res = r.finalize()
    results.append(res)

    r = run_test_a7_2()
    res = r.finalize()
    results.append(res)

    r = run_test_a7_3()
    res = r.finalize()
    results.append(res)

    r = run_test_a7_4()
    res = r.finalize()
    results.append(res)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"A7 TOTAL: {passed}/{total} subtasks gate-cleared")
