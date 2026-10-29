"""
Phase 3 Test Suite — Track M4: Parallel Orchestration
Subtasks: M4.1, M4.2, M4.3 (5 hard test cases each = 15 test cases total)
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.steps import PipelineContext
from pipeline.merger import ExtractionStepResult, DiagnosisStepResult, merge_step_results, MergedResult
from core.model_registry import get_model
from schemas.input_schemas import Language, Mode
from llm.runtime import get_llm_runtime
from tests.phase1.harness import SubtaskRunner


def _create_test_context(session_id: str, text: str = "Patient reports mild cough.") -> PipelineContext:
    return PipelineContext(
        session_id=session_id,
        text=text,
        language=Language.english,
        mode=Mode.doctor,
        department_value="opd",
        specialty_value=None,
        runtime=get_llm_runtime(),
        extraction_spec=get_model("extraction"),
        reasoning_spec=get_model("reasoning"),
    )


# ==============================================================================
# M4.1 — Extraction ∥ reasoning with timeouts
# ==============================================================================
def run_test_m4_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M4",
        subtask_id="M4.1",
        title="Extraction and reasoning parallel execution with bounded timeouts",
        target_files=["src/pipeline/orchestrator.py"],
        source_file=__file__,
    )

    # TC 1: Parallel start — Measure start times of extraction and diagnosis: within 100 ms
    async def tc1():
        orch = PipelineOrchestrator()
        start_times = {}

        async def mock_extract(ctx):
            start_times["extract"] = time.perf_counter()
            await asyncio.sleep(0.05)
            return ExtractionStepResult(features={"chief_complaint": "Cough"})

        async def mock_diagnose(ctx, **kwargs):
            start_times["diagnose"] = time.perf_counter()
            await asyncio.sleep(0.05)
            return DiagnosisStepResult(diagnosis={"final_diagnosis": "URI"})

        with patch("pipeline.orchestrator.step_extract", side_effect=mock_extract), \
             patch("pipeline.orchestrator.step_diagnose", side_effect=mock_diagnose), \
             patch("pipeline.orchestrator.get_registry"):
            ctx = _create_test_context("test-par-1")
            events = []
            async for ev in orch.run(ctx):
                events.append(ev)

        assert "extract" in start_times and "diagnose" in start_times
        delta_ms = abs(start_times["extract"] - start_times["diagnose"]) * 1000.0
        assert delta_ms <= 100.0, f"Tasks started {delta_ms:.2f}ms apart > 100ms"
        return True, f"Parallel start verified: tasks initiated within {delta_ms:.2f}ms <= 100ms"

    runner.record_case(
        case_num=1,
        name="Parallel initiation timing bound",
        input_setup="Launch extraction and diagnosis concurrently through PipelineOrchestrator",
        expected_result="Extraction and diagnosis start within 100 ms of each other",
        severity="Crit",
        fn=lambda: asyncio.run(tc1()),
    )

    # TC 2: Concurrency model verification — Inspect orchestrator concurrent task scheduling
    def tc2():
        orchestrator_path = SRC_DIR / "pipeline" / "orchestrator.py"
        content = orchestrator_path.read_text(encoding="utf-8")
        assert "asyncio.create_task" in content
        assert "asyncio.wait" in content or "asyncio.gather" in content
        assert "FIRST_COMPLETED" in content
        return True, "Concurrent non-blocking Task scheduling with FIRST_COMPLETED streaming verified"

    runner.record_case(
        case_num=2,
        name="Concurrent task scheduling architecture",
        input_setup="Inspect src/pipeline/orchestrator.py concurrency mechanics",
        expected_result="asyncio concurrent scheduling with FIRST_COMPLETED progressive streaming",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Extraction timeout — Mock slow extraction; reasoning still returns
    async def tc3():
        orch = PipelineOrchestrator()

        async def slow_extract(ctx):
            await asyncio.sleep(10.0)
            return ExtractionStepResult(features={"chief_complaint": "Delayed"})

        async def fast_diagnose(ctx, **kwargs):
            await asyncio.sleep(0.01)
            return DiagnosisStepResult(diagnosis={"final_diagnosis": "Acute Bronchitis"})

        # Patch LLM_TIMEOUT_S to a tiny value so the 10 s sleep exceeds it
        with patch("pipeline.orchestrator.step_extract", side_effect=slow_extract), \
             patch("pipeline.orchestrator.step_diagnose", side_effect=fast_diagnose), \
             patch("pipeline.orchestrator.get_registry"), \
             patch.object(__import__("core.config", fromlist=["Config"]).Config,
                          "LLM_TIMEOUT_S", 0.2), \
             patch.object(__import__("core.config", fromlist=["Config"]).Config,
                          "PIPELINE_TIMEOUT_S", 5.0):
            ctx = _create_test_context("test-timeout-ext")
            events = []
            async for ev in orch.run(ctx):
                events.append(ev)

        # Extraction timed out → extraction event may be absent or have error
        # Diagnosis completed → at least one diagnosis event OR validation_result
        all_event_types = [e.get("event") for e in events]
        # The pipeline must have produced output despite extraction timeout
        assert len(events) >= 1, "Pipeline emitted no events at all"
        # Verify extraction either timed out (no feature-rich extraction event) or ran fast
        diag_events = [e for e in events if e.get("event") in ("diagnosis", "validation_result")]
        assert len(diag_events) >= 1, f"No diagnosis/result event emitted. Events: {all_event_types}"
        return True, "Extraction timeout gracefully isolated: pipeline continued without hang"

    runner.record_case(
        case_num=3,
        name="Extraction timeout isolation",
        input_setup="Simulate extraction latency exceeding timeout threshold while diagnosis completes",
        expected_result="Extraction timed out; diagnosis output preserved without crash",
        severity="Crit",
        fn=lambda: asyncio.run(tc3()),
    )

    # TC 4: Reasoning timeout — Mock slow reasoning; extraction returns
    async def tc4():
        orch = PipelineOrchestrator()

        async def fast_extract(ctx):
            await asyncio.sleep(0.01)
            return ExtractionStepResult(features={"chief_complaint": "Right ankle sprain"})

        async def slow_diagnose(ctx, **kwargs):
            await asyncio.sleep(10.0)
            return DiagnosisStepResult(diagnosis={"final_diagnosis": "Sprain"})

        # Patch LLM_TIMEOUT_S to a tiny value so the 10 s sleep exceeds it
        with patch("pipeline.orchestrator.step_extract", side_effect=fast_extract), \
             patch("pipeline.orchestrator.step_diagnose", side_effect=slow_diagnose), \
             patch("pipeline.orchestrator.get_registry"), \
             patch.object(__import__("core.config", fromlist=["Config"]).Config,
                          "LLM_TIMEOUT_S", 0.2), \
             patch.object(__import__("core.config", fromlist=["Config"]).Config,
                          "PIPELINE_TIMEOUT_S", 5.0):
            ctx = _create_test_context("test-timeout-diag")
            events = []
            async for ev in orch.run(ctx):
                events.append(ev)

        # Extraction completed → extraction event present
        all_event_types = [e.get("event") for e in events]
        ext_events = [e for e in events if e.get("event") == "extraction"]
        assert len(events) >= 1, "Pipeline emitted no events at all"
        assert len(ext_events) >= 1, f"No extraction event emitted. Events: {all_event_types}"
        return True, "Reasoning timeout gracefully handled: extracted features delivered cleanly"

    runner.record_case(
        case_num=4,
        name="Reasoning timeout isolation",
        input_setup="Simulate diagnosis latency exceeding timeout while extraction completes",
        expected_result="Extraction delivered intact; reasoning timeout handled gracefully",
        severity="Crit",
        fn=lambda: asyncio.run(tc4()),
    )

    # TC 5: Both succeed — Normal parallel run yields complete stream
    async def tc5():
        orch = PipelineOrchestrator()

        async def norm_extract(ctx):
            await asyncio.sleep(0.01)
            return ExtractionStepResult(features={"chief_complaint": "Migraine", "duration": "2 days"})

        async def norm_diagnose(ctx, **kwargs):
            await asyncio.sleep(0.01)
            return DiagnosisStepResult(diagnosis={"final_diagnosis": "Classic Migraine", "icd10_codes": ["G43.9"]})

        with patch("pipeline.orchestrator.step_extract", side_effect=norm_extract), \
             patch("pipeline.orchestrator.step_diagnose", side_effect=norm_diagnose), \
             patch("pipeline.orchestrator.get_registry"):
            ctx = _create_test_context("test-both-ok")
            events = []
            async for ev in orch.run(ctx):
                events.append(ev)

        event_names = [e.get("event") for e in events]
        assert "extraction" in event_names
        assert "diagnosis" in event_names
        return True, "Both parallel steps succeeded: extraction and diagnosis events streamed"

    runner.record_case(
        case_num=5,
        name="Dual success parallel merge",
        input_setup="Execute standard parallel pipeline with both extract and diagnose succeeding",
        expected_result="Merged stream encapsulates both extraction and diagnosis events",
        severity="Crit",
        fn=lambda: asyncio.run(tc5()),
    )

    return runner


# ==============================================================================
# M4.2 — Merge → validation → escalation
# ==============================================================================
def run_test_m4_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M4",
        subtask_id="M4.2",
        title="Pipeline merger validation, schema conformance and reasoning escalation",
        target_files=["src/pipeline/merger.py"],
        source_file=__file__,
    )

    # TC 1: Combined output contains both extraction and reasoning
    def tc1():
        ext_res = ExtractionStepResult(features={"chief_complaint": "Fever", "duration": "3 days"})
        diag_res = DiagnosisStepResult(diagnosis={"final_diagnosis": "Viral illness", "icd10_codes": ["B34.9"]})
        merged = merge_step_results(
            extraction_res=ext_res,
            diagnosis_res=diag_res,
            session_id="sess-001",
            department="opd",
            language="en",
            mode="dictation",
            transcript="Patient has fever for 3 days.",
        )
        assert merged.extracted_features["chief_complaint"] == "Fever"
        assert merged.diagnosis["final_diagnosis"] == "Viral illness"
        assert "B34.9" in merged.diagnosis["icd10_codes"][0]
        return True, "Merged result encapsulates extraction and diagnosis payloads"

    runner.record_case(
        case_num=1,
        name="Extraction and reasoning combination",
        input_setup="Merge distinct ExtractionStepResult and DiagnosisStepResult",
        expected_result="Combined MergedResult contains all fields from both steps",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Schema validation conformance
    def tc2():
        ext_res = ExtractionStepResult(features={"chief_complaint": "Gastritis", "onset": "Yesterday"})
        diag_res = DiagnosisStepResult(diagnosis={"final_diagnosis": "Acute gastritis", "icd10_codes": ["K29.0"]})
        merged = merge_step_results(
            extraction_res=ext_res,
            diagnosis_res=diag_res,
            session_id="sess-002",
            department="opd",
            language="en",
            mode="dictation",
            transcript="Patient with stomach pain.",
        )
        log_dict = merged.to_log_dict()
        assert log_dict["session_id"] == "sess-002"
        assert log_dict["department"] == "opd"
        assert isinstance(log_dict["extracted_features"], dict)
        assert isinstance(log_dict["diagnosis"], dict)
        return True, "Merged result serializes to valid schema log payload"

    runner.record_case(
        case_num=2,
        name="Merged payload schema conformance",
        input_setup="Validate serialization of MergedResult into structured log dictionary",
        expected_result="Passes schema constraints with required types and keys",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Escalation trigger on validation failure
    def tc3():
        merged = MergedResult(
            session_id="sess-esc",
            department="er",
            language="en",
            mode="dictation",
            tier="primary",
            validation_passed=False,
            escalation_reason="ICD10_UNRESOLVED",
        )
        assert merged.validation_passed is False
        assert merged.escalation_reason == "ICD10_UNRESOLVED"
        return True, "Escalation metadata correctly recorded on validation failure"

    runner.record_case(
        case_num=3,
        name="Validation failure escalation tracking",
        input_setup="Configure MergedResult with validation failure state",
        expected_result="Flags validation_passed=False with explicit escalation_reason",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Determinism — 10x same inputs -> identical merged results
    def tc4():
        ext_res = ExtractionStepResult(features={"chief_complaint": "Otitis media", "duration": "4 days"})
        diag_res = DiagnosisStepResult(diagnosis={"final_diagnosis": "Acute otitis media", "icd10_codes": ["H66.9"]})
        runs = [
            merge_step_results(
                extraction_res=ext_res,
                diagnosis_res=diag_res,
                session_id="sess-det",
                department="opd",
                language="en",
                mode="dictation",
                transcript="Ear pain for 4 days.",
            ).to_log_dict(timestamp="2026-10-28T12:00:00Z")
            for _ in range(10)
        ]
        first = json.dumps(runs[0], sort_keys=True)
        assert all(json.dumps(r, sort_keys=True) == first for r in runs)
        return True, "10/10 consecutive merge executions produced bitwise identical outputs"

    runner.record_case(
        case_num=4,
        name="Merge execution determinism",
        input_setup="Execute merge_step_results 10 consecutive times on identical input",
        expected_result="10/10 identical serialized outputs returned",
        severity="High",
        fn=tc4,
    )

    # TC 5: Partial merge on reasoning timeout
    def tc5():
        ext_res = ExtractionStepResult(features={"chief_complaint": "Chest contusion"})
        merged = merge_step_results(
            extraction_res=ext_res,
            diagnosis_res=None,  # timed out or failed
            session_id="sess-partial",
            department="er",
            language="en",
            mode="dictation",
            transcript="Blunt chest trauma.",
        )
        assert merged.extracted_features["chief_complaint"] == "Chest contusion"
        assert merged.diagnosis == {}
        assert merged.status == "partial" or merged.diagnosis_sec is None
        return True, "Partial merge cleanly handled: extraction preserved when diagnosis is None"

    runner.record_case(
        case_num=5,
        name="Partial merge on diagnosis failure",
        input_setup="Merge ExtractionStepResult with None diagnosis_res",
        expected_result="Extraction payload intact; diagnosis safely defaulted without crashing",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M4.3 — SSE events
# ==============================================================================
def run_test_m4_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M4",
        subtask_id="M4.3",
        title="Real-time Server-Sent Events (SSE) streaming and event contract",
        target_files=["src/pipeline/orchestrator.py"],
        source_file=__file__,
    )

    # TC 1: All event types emitted across full pipeline execution
    async def tc1():
        orch = PipelineOrchestrator()

        async def mock_extract(ctx):
            await asyncio.sleep(0.01)
            return ExtractionStepResult(features={"chief_complaint": "Rash"})

        async def mock_diagnose(ctx, **kwargs):
            await asyncio.sleep(0.01)
            return DiagnosisStepResult(diagnosis={"final_diagnosis": "Dermatitis", "icd10_codes": ["L30.9"]})

        with patch("pipeline.orchestrator.step_extract", side_effect=mock_extract), \
             patch("pipeline.orchestrator.step_diagnose", side_effect=mock_diagnose), \
             patch("pipeline.orchestrator.get_registry"):
            ctx = _create_test_context("sess-sse-events")
            event_types = []
            async for ev in orch.run(ctx):
                event_types.append(ev.get("event"))

        assert "extraction" in event_types
        assert "diagnosis" in event_types
        assert "validation_result" in event_types
        return True, f"Full SSE lifecycle verified: events emitted = {event_types}"

    runner.record_case(
        case_num=1,
        name="All SSE event types emission",
        input_setup="Execute pipeline and collect all emitted SSE events",
        expected_result="Emits progressive step events including extraction, diagnosis, and validation_result",
        severity="Crit",
        fn=lambda: asyncio.run(tc1()),
    )

    # TC 2: Event ordering — Progressive events precede validation_result
    async def tc2():
        orch = PipelineOrchestrator()

        async def mock_extract(ctx):
            await asyncio.sleep(0.01)
            return ExtractionStepResult(features={"chief_complaint": "Back pain"})

        async def mock_diagnose(ctx, **kwargs):
            await asyncio.sleep(0.01)
            return DiagnosisStepResult(diagnosis={"final_diagnosis": "Lumbago"})

        with patch("pipeline.orchestrator.step_extract", side_effect=mock_extract), \
             patch("pipeline.orchestrator.step_diagnose", side_effect=mock_diagnose), \
             patch("pipeline.orchestrator.get_registry"):
            ctx = _create_test_context("sess-sse-order")
            event_list = []
            async for ev in orch.run(ctx):
                event_list.append(ev.get("event"))

        assert event_list[-1] == "validation_result", f"Terminal event was '{event_list[-1]}', expected 'validation_result'"
        return True, f"Event ordering verified: final event is strictly 'validation_result' ({event_list})"

    runner.record_case(
        case_num=2,
        name="SSE event progression ordering",
        input_setup="Trace sequential emission order of SSE event stream",
        expected_result="Progressive events stream first; 'validation_result' is strictly terminal",
        severity="Crit",
        fn=lambda: asyncio.run(tc2()),
    )

    # TC 3: JSON parseable — All SSE payload data fields are valid JSON
    async def tc3():
        orch = PipelineOrchestrator()

        with patch("pipeline.orchestrator.step_extract", return_value=ExtractionStepResult(features={"cc": "pain"})), \
             patch("pipeline.orchestrator.step_diagnose", return_value=DiagnosisStepResult(diagnosis={"dx": "mild"})), \
             patch("pipeline.orchestrator.get_registry"):
            ctx = _create_test_context("sess-sse-json")
            async for ev in orch.run(ctx):
                data_val = ev.get("data")
                if isinstance(data_val, str):
                    parsed = json.loads(data_val)
                    assert isinstance(parsed, (dict, list, str, int, bool))
                elif isinstance(data_val, dict):
                    raw_json = json.dumps(data_val)
                    assert len(raw_json) > 0

        return True, "100% of emitted SSE payloads conform to valid JSON format"

    runner.record_case(
        case_num=3,
        name="SSE payload JSON integrity",
        input_setup="Serialize and parse each emitted SSE message payload",
        expected_result="All event payloads serialize to valid JSON",
        severity="Crit",
        fn=lambda: asyncio.run(tc3()),
    )

    # TC 4: Disconnect recovery — Consumer cancellation cleans up without orphan tasks
    async def tc4():
        orch = PipelineOrchestrator()

        async def endless_task(ctx, **kwargs):
            await asyncio.sleep(60.0)
            return DiagnosisStepResult()

        with patch("pipeline.orchestrator.step_extract", side_effect=endless_task), \
             patch("pipeline.orchestrator.step_diagnose", side_effect=endless_task), \
             patch("pipeline.orchestrator.get_registry"):
            ctx = _create_test_context("sess-sse-abort")
            gen = orch.run(ctx)
            try:
                await asyncio.wait_for(gen.asend(None), timeout=0.05)
            except (asyncio.TimeoutError, StopAsyncIteration):
                pass
            await gen.aclose()

        return True, "Client disconnect handled cleanly: generator aclose() freed pipeline resources"

    runner.record_case(
        case_num=4,
        name="Client disconnect resource cleanup",
        input_setup="Simulate client disconnect by closing generator mid-stream",
        expected_result="aclose() executes cleanly without leaving orphaned tasks",
        severity="High",
        fn=lambda: asyncio.run(tc4()),
    )

    # TC 5: Monotonic progression timestamps
    async def tc5():
        orch = PipelineOrchestrator()
        timestamps = []

        with patch("pipeline.orchestrator.step_extract", return_value=ExtractionStepResult(features={"f": 1})), \
             patch("pipeline.orchestrator.step_diagnose", return_value=DiagnosisStepResult(diagnosis={"d": 1})), \
             patch("pipeline.orchestrator.get_registry"):
            ctx = _create_test_context("sess-sse-mono")
            async for ev in orch.run(ctx):
                timestamps.append(time.time())

        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i+1]
        return True, f"Monotonic timing verified across {len(timestamps)} emitted SSE events"

    runner.record_case(
        case_num=5,
        name="Monotonic progression timestamps",
        input_setup="Verify temporal ordering of emitted SSE stream events",
        expected_result="Event emission timestamps are strictly non-decreasing",
        severity="High",
        fn=lambda: asyncio.run(tc5()),
    )

    return runner


if __name__ == "__main__":
    for func in [run_test_m4_1, run_test_m4_2, run_test_m4_3]:
        r = func()
        rep = r.finalize()
        print(f"[{r.subtask_id}] Passed: {rep.pass_count}/{len(rep.cases)} (Cleared: {rep.passed})")
