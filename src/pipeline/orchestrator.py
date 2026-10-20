"""Core Pipeline Orchestrator.

Extracts the full post-transcription pipeline into an event-streaming orchestrator.
Yields raw event dicts: {"event": str, "data": dict | str}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Callable, Coroutine, Dict, List, Optional

from core.config import Config
from core.model_registry import get_model
from validation.pipeline import ClinicalValidationPipeline
from pipeline.merger import (
    DiagnosisStepResult,
    ExtractionStepResult,
    MergedResult,
    merge_step_results,
)
from pipeline.steps import (
    PipelineContext,
    step_diagnose,
    step_extract,
    step_red_flags,
    step_refine_stream,
    step_translate_stream,
)

logger = logging.getLogger(__name__)

# SSE event name emitted when all reasoning models fail — the UI shows
# a graceful "reasoning pending" message instead of crashing.
_REASONING_PENDING_EVENT = "reasoning_pending"


class PipelineOrchestrator:
    """Stateless orchestrator that drives clinical pipeline execution."""

    def __init__(
        self,
        on_complete: Optional[Callable[[MergedResult], Coroutine[Any, Any, None]]] = None,
    ) -> None:
        """Initialize orchestrator with optional async completion callback."""
        self._on_complete = on_complete

    async def run(
        self,
        ctx: PipelineContext,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute all pipeline steps and stream SSE event dictionaries."""
        extraction_res: Optional[ExtractionStepResult] = None
        diagnosis_res: Optional[DiagnosisStepResult] = None
        parallel_tasks: List[asyncio.Task] = []
        validation_dict: Optional[Dict[str, Any]] = None
        validation_passed: bool = True
        current_tier: str = "primary"
        escalation_reason: Optional[str] = None

        try:
            # ── 1. Refinement Step ─────────────────────────────────────────
            async for ev in step_refine_stream(ctx):
                yield ev
                if ev.get("event") == "error":
                    return

            # ── 2. Translation Step (Arabic only) ──────────────────────────
            async for ev in step_translate_stream(ctx):
                yield ev
                if ev.get("event") == "error":
                    return

            # Text handed to extraction & diagnosis
            ctx.patient_story = ctx.translated_text
            routing_signal = step_red_flags(ctx)
            yield {"event": "routing_signal", "data": routing_signal}

            # ── 3. Parallel Extraction & Diagnosis (with shared timeout) ──
            timeout_s = getattr(Config, "LLM_TIMEOUT_S", 45.0)

            async def _run_extract() -> ExtractionStepResult:
                return await step_extract(ctx)

            async def _run_diagnose(spec: Optional[Any] = None) -> DiagnosisStepResult:
                return await step_diagnose(ctx, reasoning_spec=spec)

            extract_task = asyncio.create_task(_run_extract())
            diagnose_task = asyncio.create_task(_run_diagnose(ctx.reasoning_spec))
            parallel_tasks = [extract_task, diagnose_task]

            try:
                extraction_res, diagnosis_res = await asyncio.wait_for(
                    asyncio.gather(extract_task, diagnose_task),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error("[Pipeline] Extraction || Diagnosis timed out after %ss", timeout_s)
                if not extract_task.done():
                    extract_task.cancel()
                    extraction_res = ExtractionStepResult(error=f"Timeout after {timeout_s}s")
                else:
                    try:
                        if extract_task.cancelled():
                            extraction_res = ExtractionStepResult(error=f"Timeout after {timeout_s}s")
                        else:
                            extraction_res = extract_task.result()
                    except asyncio.CancelledError:
                        extraction_res = ExtractionStepResult(error=f"Timeout after {timeout_s}s")
                    except Exception as exc:
                        extraction_res = ExtractionStepResult(error=str(exc))

                if not diagnose_task.done():
                    diagnose_task.cancel()
                    diagnosis_res = DiagnosisStepResult(error=f"Timeout after {timeout_s}s")
                else:
                    try:
                        if diagnose_task.cancelled():
                            diagnosis_res = DiagnosisStepResult(error=f"Timeout after {timeout_s}s")
                        else:
                            diagnosis_res = diagnose_task.result()
                    except asyncio.CancelledError:
                        diagnosis_res = DiagnosisStepResult(error=f"Timeout after {timeout_s}s")
                    except Exception as exc:
                        diagnosis_res = DiagnosisStepResult(error=str(exc))

            # ── Emit extraction event ──────────────────────────────────────
            if extraction_res.error:
                event_data = {
                    "department": ctx.department_value,
                    "model": ctx.extraction_spec.label,
                    "error": extraction_res.error,
                }
            else:
                event_data = {
                    "department": ctx.department_value,
                    "patient_story": ctx.patient_story,
                    "routing_signal": ctx.routing_signal,
                    "extracted_features": extraction_res.features,
                    "extraction_sec": extraction_res.extraction_sec,
                    "questions": extraction_res.questions,
                    "question_generation_sec": extraction_res.question_generation_sec,
                    "model": ctx.extraction_spec.label,
                    "usage": extraction_res.meta,
                }
            yield {"event": "extraction", "data": json.dumps(event_data)}

            # ── Emit diagnosis event ───────────────────────────────────────
            def _build_diagnosis_event(res: DiagnosisStepResult, model_label: str) -> Dict[str, Any]:
                if res.error:
                    return {
                        "department": ctx.department_value,
                        "routing_signal": ctx.routing_signal,
                        "model": model_label,
                        "error": res.error,
                        "history_used": bool(ctx.patient_history_text),
                        "history_visit_count": len(ctx.patient_history_visits),
                        "history_error": ctx.history_error,
                    }
                return {
                    "department": ctx.department_value,
                    "diagnosis": res.diagnosis,
                    "diagnosis_sec": res.diagnosis_sec,
                    "model": model_label,
                    "usage": res.meta,
                    "history_used": bool(ctx.patient_history_text),
                    "history_visit_count": len(ctx.patient_history_visits),
                    "history_check": res.history_check,
                    "history_error": ctx.history_error,
                }

            yield {"event": "diagnosis", "data": json.dumps(_build_diagnosis_event(diagnosis_res, ctx.reasoning_spec.label))}

            # ── 4. M3 Validation & Escalation (Deliverables 2.2, 2.3, 2.4) ──
            raw_extract_data = extraction_res.features if (extraction_res and not extraction_res.error) else {}
            raw_diag_data = diagnosis_res.diagnosis if (diagnosis_res and not diagnosis_res.error) else {}

            validation_report = ClinicalValidationPipeline.validate(
                department=ctx.department_value,
                raw_extraction=raw_extract_data,
                raw_diagnosis=raw_diag_data,
                patient_history=ctx.patient_history_visits,
                translated_text=ctx.patient_story,
            )

            validation_dict = validation_report.to_dict()
            validation_passed = validation_report.is_valid
            current_tier = "primary"
            escalation_reason: Optional[str] = None

            if validation_passed:
                yield {
                    "event": "validation_result",
                    "data": json.dumps({
                        "is_valid": True,
                        "tier": "primary",
                        "cautions": validation_dict.get("cautions", []),
                        "missing_fields": validation_dict.get("missing_fields", {}),
                        "validation_latency_ms": validation_dict.get("metrics", {}).get("validation_latency_ms", 0.0),
                        "schema_rejection": None,
                    }),
                }
            else:
                # Validation rejected -> Trigger B4.2 one-time router escalation to Sol model
                rejection_msg = "Validation rejected: schema or ICD-10 resolution failure"
                if validation_report.schema_rejection:
                    rejection_msg = f"Schema rejection: {validation_report.schema_rejection.reason}"
                escalation_reason = rejection_msg
                current_tier = "escalated"

                logger.warning(
                    "[Pipeline] Validation failure (%s). Escalating reasoning to higher-tier Sol model.",
                    escalation_reason,
                )

                sol_spec = get_model("reasoning", escalation=True)
                escalated_diagnose_task = asyncio.create_task(_run_diagnose(sol_spec))
                parallel_tasks.append(escalated_diagnose_task)

                try:
                    escalated_res = await asyncio.wait_for(
                        escalated_diagnose_task,
                        timeout=timeout_s,
                    )
                    diagnosis_res = escalated_res
                except asyncio.TimeoutError:
                    logger.error("[Pipeline] Escalated diagnosis timed out after %ss", timeout_s)
                    diagnosis_res = DiagnosisStepResult(error=f"Timeout after {timeout_s}s")
                except Exception as exc:
                    logger.error("[Pipeline] Escalated diagnosis error: %s", exc)
                    diagnosis_res = DiagnosisStepResult(error=str(exc))

                # Stream updated diagnosis from escalated tier
                yield {"event": "diagnosis", "data": json.dumps(_build_diagnosis_event(diagnosis_res, sol_spec.label))}

                # If escalated diagnosis ALSO failed — emit reasoning_pending and
                # skip re-validation so the UI degrades gracefully without crashing.
                if diagnosis_res.error:
                    logger.error(
                        "[Pipeline] All reasoning models failed (primary + escalated Sol). "
                        "Emitting %s event.", _REASONING_PENDING_EVENT
                    )
                    yield {
                        "event": _REASONING_PENDING_EVENT,
                        "data": json.dumps({
                            "message": "Diagnosis reasoning could not be completed at this time. "
                                       "The clinical note extraction is available.",
                            "primary_error": escalation_reason,
                            "fallback_error": diagnosis_res.error,
                        }),
                    }
                    # Skip re-validation — return now, on_complete will still persist the session
                    return

                # Re-validate with escalated diagnosis output
                raw_diag_data_esc = diagnosis_res.diagnosis if (diagnosis_res and not diagnosis_res.error) else {}
                revalidation_report = ClinicalValidationPipeline.validate(
                    department=ctx.department_value,
                    raw_extraction=raw_extract_data,
                    raw_diagnosis=raw_diag_data_esc,
                    patient_history=ctx.patient_history_visits,
                    translated_text=ctx.patient_story,
                )
                validation_report = revalidation_report
                validation_dict = revalidation_report.to_dict()
                validation_passed = revalidation_report.is_valid

                yield {
                    "event": "validation_result",
                    "data": json.dumps({
                        "is_valid": validation_passed,
                        "tier": "escalated",
                        "model": sol_spec.label,
                        "escalation_reason": escalation_reason,
                        "cautions": validation_dict.get("cautions", []),
                        "missing_fields": validation_dict.get("missing_fields", {}),
                        "validation_latency_ms": validation_dict.get("metrics", {}).get("validation_latency_ms", 0.0),
                        "schema_rejection": validation_dict.get("schema_rejection"),
                    }),
                }

        finally:
            for task in parallel_tasks:
                if not task.done():
                    task.cancel()
            if parallel_tasks:
                await asyncio.gather(*parallel_tasks, return_exceptions=True)

            total_sec = round(time.perf_counter() - ctx.pipeline_start, 3)
            final_reasoning_model = ctx.reasoning_spec.id
            if current_tier == "escalated":
                try:
                    final_reasoning_model = get_model("reasoning", escalation=True).id
                except Exception:
                    final_reasoning_model = "gpt-5.6-sol"

            merged = merge_step_results(
                extraction_res=extraction_res,
                diagnosis_res=diagnosis_res,
                session_id=ctx.session_id,
                department=ctx.department_value,
                language=ctx.language.value,
                mode=ctx.mode.value,
                transcript=ctx.text,
                refined_text=ctx.refined_text,
                translated_text=ctx.translated_text,
                patient_story=ctx.patient_story,
                routing_signal=ctx.routing_signal,
                specialty=ctx.specialty_value,
                patient_id=ctx.patient_id,
                business_unit=ctx.business_unit_value,
                use_history=ctx.use_history,
                history_visit_count=len(ctx.patient_history_visits),
                history_error=ctx.history_error,
                extraction_model=ctx.extraction_spec.id,
                reasoning_model=final_reasoning_model,
                refinement_sec=ctx.refinement_sec,
                translation_sec=ctx.translation_sec,
                transcription_sec=ctx.transcription_sec,
                total_sec=total_sec,
                validation_report=validation_dict,
                validation_passed=validation_passed,
                tier=current_tier,
                escalation_reason=escalation_reason,
            )

            if self._on_complete:
                try:
                    await self._on_complete(merged)
                except Exception as exc:
                    logger.error("[Pipeline] on_complete callback failed: %s", exc)

