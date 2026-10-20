"""
tests/test_pipeline.py
======================
Unit tests for Phase 2: True Parallel Orchestration (M4.1 + M4.2).

Deliverables verified:
2.1 Parallel gather with shared timeout — parallel execution time ≈ max(extraction_time, diagnosis_time)
2.2 Merge outputs -> M3 validation — combined extraction + diagnosis into validation/pipeline.py
2.3 One router call on failure (B4.2) — escalation to Sol model when M3 validation rejects
2.4 SSE event: validation_result — emitted with pass/fail + cautions
2.5 Test gate: unit test
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from core.config import Config
from core.model_registry import ModelSpec
from pipeline.merger import DiagnosisStepResult, ExtractionStepResult, MergedResult
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.steps import PipelineContext
from schemas.input_schemas import Language, Mode


def _make_dummy_context() -> PipelineContext:
    luna_spec = ModelSpec("gpt-5.6-luna", "GPT 5.6 Luna", "gpt-5.6-luna", True)
    return PipelineContext(
        session_id="test-session-phase2",
        text="Patient presents with left knee pain after slipping.",
        language=Language.english,
        mode=Mode.doctor,
        department_value="opd",
        specialty_value="Orthopedics",
        runtime=MagicMock(),
        extraction_spec=luna_spec,
        reasoning_spec=luna_spec,
    )


@pytest.mark.asyncio
async def test_parallel_execution_timing_max_not_sum():
    """Verify parallel execution time is approximately max(extract, diagnose) not sum."""
    from validation.pipeline import ClinicalValidationPipeline

    # Warm-up ICD-10 registry singleton and validation rules
    ClinicalValidationPipeline.validate(
        department="opd",
        raw_extraction={"chief_complaint": "test"},
        raw_diagnosis={"icd10_codes": ["M17.12"]},
    )

    ctx = _make_dummy_context()

    async def mock_extract(c: PipelineContext) -> ExtractionStepResult:
        await asyncio.sleep(0.15)
        return ExtractionStepResult(
            features={"chief_complaint": "left knee pain", "plan_of_care": "rest"},
            extraction_sec=0.15,
        )

    async def mock_diagnose(c: PipelineContext, reasoning_spec: Optional[ModelSpec] = None) -> DiagnosisStepResult:
        await asyncio.sleep(0.15)
        return DiagnosisStepResult(
            diagnosis={
                "icd10_codes": ["M17.12 - Unilateral primary osteoarthritis, left knee"],
                "reasoning": "Left knee pain supports osteoarthritis",
            },
            diagnosis_sec=0.15,
        )

    with patch("pipeline.orchestrator.step_extract", side_effect=mock_extract), \
         patch("pipeline.orchestrator.step_diagnose", side_effect=mock_diagnose):

        orchestrator = PipelineOrchestrator()
        t0 = time.perf_counter()
        events = []
        async for ev in orchestrator.run(ctx):
            events.append(ev)
        elapsed = time.perf_counter() - t0

        event_names = [e["event"] for e in events]
        assert "extraction" in event_names
        assert "diagnosis" in event_names
        assert "validation_result" in event_names

        # Max is 0.15s; sum is 0.30s. Parallel execution must be well under sum (< 0.28s)
        assert elapsed < 0.28, f"Execution took {elapsed:.3f}s; expected parallel execution (< 0.28s)"


@pytest.mark.asyncio
async def test_timeout_cancellation_and_error_handling():
    """Verify shared timeout cancels hanging tasks and records timeout errors."""
    ctx = _make_dummy_context()

    async def mock_hanging_extract(c: PipelineContext) -> ExtractionStepResult:
        await asyncio.sleep(5.0)
        return ExtractionStepResult(features={})

    async def mock_fast_diagnose(c: PipelineContext, reasoning_spec: Optional[ModelSpec] = None) -> DiagnosisStepResult:
        await asyncio.sleep(0.02)
        return DiagnosisStepResult(
            diagnosis={"icd10_codes": ["M17.12"]},
            diagnosis_sec=0.02,
        )

    with patch.object(Config, "LLM_TIMEOUT_S", 0.08), \
         patch("pipeline.orchestrator.step_extract", side_effect=mock_hanging_extract), \
         patch("pipeline.orchestrator.step_diagnose", side_effect=mock_fast_diagnose):

        orchestrator = PipelineOrchestrator()
        events = []
        async for ev in orchestrator.run(ctx):
            events.append(ev)

        extract_events = [e for e in events if e["event"] == "extraction"]
        assert len(extract_events) == 1
        data = json.loads(extract_events[0]["data"])
        assert "Timeout after" in data.get("error", "")


@pytest.mark.asyncio
async def test_m3_validation_pass_and_sse_event():
    """Verify valid clinical output generates SSE validation_result with tier=primary."""
    ctx = _make_dummy_context()

    valid_extract = ExtractionStepResult(
        features={"chief_complaint": "Left knee pain", "plan_of_care": "Rest and ice"},
        extraction_sec=0.05,
    )
    valid_diag = DiagnosisStepResult(
        diagnosis={
            "icd10_codes": ["M17.12 - Unilateral primary osteoarthritis, left knee"],
            "reasoning": "Left knee pain examination matches osteoarthritis",
        },
        diagnosis_sec=0.05,
    )

    with patch("pipeline.orchestrator.step_extract", new=AsyncMock(return_value=valid_extract)), \
         patch("pipeline.orchestrator.step_diagnose", new=AsyncMock(return_value=valid_diag)):

        orchestrator = PipelineOrchestrator()
        events = []
        async for ev in orchestrator.run(ctx):
            events.append(ev)

        val_events = [e for e in events if e["event"] == "validation_result"]
        assert len(val_events) == 1
        val_data = json.loads(val_events[0]["data"])
        assert val_data["is_valid"] is True
        assert val_data["tier"] == "primary"
        assert "cautions" in val_data


@pytest.mark.asyncio
async def test_escalation_to_sol_on_validation_failure():
    """Verify B4.2: validation failure triggers one escalation to Sol model."""
    ctx = _make_dummy_context()

    # Rejection: hallucinated / invalid ICD-10 code
    failed_diag = DiagnosisStepResult(
        diagnosis={
            "icd10_codes": ["FAKE999.00 - Hallucinated condition"],
            "reasoning": "Unsupported code",
        },
        diagnosis_sec=0.04,
    )
    # Sol escalated model provides valid code
    sol_diag = DiagnosisStepResult(
        diagnosis={
            "icd10_codes": ["M17.12 - Unilateral primary osteoarthritis, left knee"],
            "reasoning": "Escalated model identified canonical diagnosis",
        },
        diagnosis_sec=0.05,
    )

    mock_step_diagnose = AsyncMock(side_effect=[failed_diag, sol_diag])

    valid_extract = ExtractionStepResult(
        features={"chief_complaint": "Left knee pain", "plan_of_care": "Ice pack"},
        extraction_sec=0.04,
    )

    callback_record = []
    async def on_complete_hook(merged: MergedResult):
        callback_record.append(merged)

    with patch("pipeline.orchestrator.step_extract", new=AsyncMock(return_value=valid_extract)), \
         patch("pipeline.orchestrator.step_diagnose", mock_step_diagnose):

        orchestrator = PipelineOrchestrator(on_complete=on_complete_hook)
        events = []
        async for ev in orchestrator.run(ctx):
            events.append(ev)

        # Diagnose was called twice: once for primary (Luna) and once for escalated (Sol)
        assert mock_step_diagnose.call_count == 2
        # Check that second call was passed Sol model spec
        second_call_spec = mock_step_diagnose.call_args_list[1].kwargs.get("reasoning_spec")
        assert second_call_spec is not None
        assert second_call_spec.id == "gpt-5.6-sol"

        # Check validation_result event
        val_events = [e for e in events if e["event"] == "validation_result"]
        assert len(val_events) == 1
        val_data = json.loads(val_events[0]["data"])
        assert val_data["tier"] == "escalated"
        assert val_data["model"] == "GPT 5.6 Sol"

        # Check final MergedResult handed to on_complete
        assert len(callback_record) == 1
        merged = callback_record[0]
        assert merged.tier == "escalated"
        assert merged.reasoning_model == "gpt-5.6-sol"
        assert "Validation rejected" in (merged.escalation_reason or "")
