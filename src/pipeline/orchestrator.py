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
from pipeline.routing import (
    RoutingDecision,
    classify_complexity,
    decision_without_router,
    fallback_deployment_for_tier,
    failure_tier,
    spec_for_tier,
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

            # ── 3. Parallel Extraction & Reasoning (with shared timeout) ──
            # Extraction always starts immediately. Reasoning starts on the
            # routed tier: when ROUTER_ENABLED=true the router runs concurrently
            # with extraction and the tier is chosen when it returns; otherwise
            # reasoning starts at once on the simple tier.
            timeout_s = getattr(Config, "LLM_TIMEOUT_S", 45.0)

            async def _run_extract() -> ExtractionStepResult:
                return await step_extract(ctx)

            async def _run_diagnose(spec: Optional[Any] = None, fallback: Optional[str] = None) -> DiagnosisStepResult:
                return await step_diagnose(ctx, reasoning_spec=spec, fallback_deployment=fallback)

            extract_task = asyncio.create_task(_run_extract())
            parallel_tasks = [extract_task]

            routing_decision: Optional[RoutingDecision] = None
            if Config.router_enabled():
                router_task = asyncio.create_task(
                    classify_complexity(ctx.runtime, ctx.patient_story, ctx.patient_history_text)
                )
                parallel_tasks.append(router_task)
                try:
                    routing_decision = await asyncio.wait_for(router_task, timeout=timeout_s)
                except asyncio.TimeoutError:
                    logger.error("[Pipeline] Router timed out after %ss; using failure tier", timeout_s)
                    routing_decision = RoutingDecision(
                        complexity=failure_tier(),
                        tier=failure_tier(),
                        router_deployment=Config.foundry_model_router(),
                        router_ok=False,
                    )
                except Exception as exc:
                    logger.error("[Pipeline] Router error: %s; using failure tier", exc)
                    routing_decision = RoutingDecision(
                        complexity=failure_tier(),
                        tier=failure_tier(),
                        router_deployment=Config.foundry_model_router(),
                        router_ok=False,
                    )
            else:
                routing_decision = decision_without_router()

            ctx.routing_decision = routing_decision
            reasoning_spec = spec_for_tier(routing_decision.tier)
            reasoning_fallback = fallback_deployment_for_tier(routing_decision.tier)
            diagnose_task = asyncio.create_task(_run_diagnose(reasoning_spec, reasoning_fallback))
            parallel_tasks.append(diagnose_task)

            # ── Emit each parallel result as soon as it completes ──────────
            # The extraction and reasoning branches run concurrently; whichever
            # finishes first is streamed immediately so the UI does not wait for
            # the slower branch. A shared deadline bounds the whole pair.
            def _served(meta: Optional[Dict[str, Any]], default: str) -> str:
                if isinstance(meta, dict) and meta.get("served_deployment"):
                    return str(meta["served_deployment"])
                return default

            def _reasoning_label() -> str:
                if diagnosis_res is not None:
                    return _served(diagnosis_res.meta, reasoning_spec.foundry_deployment)
                return reasoning_spec.foundry_deployment

            def _extraction_result_from(task: "asyncio.Task") -> ExtractionStepResult:
                try:
                    if task.cancelled():
                        return ExtractionStepResult(error=f"Timeout after {timeout_s}s")
                    return task.result()
                except asyncio.CancelledError:
                    return ExtractionStepResult(error=f"Timeout after {timeout_s}s")
                except Exception as exc:
                    return ExtractionStepResult(error=str(exc))

            def _diagnosis_result_from(task: "asyncio.Task") -> DiagnosisStepResult:
                try:
                    if task.cancelled():
                        return DiagnosisStepResult(error=f"Timeout after {timeout_s}s")
                    return task.result()
                except asyncio.CancelledError:
                    return DiagnosisStepResult(error=f"Timeout after {timeout_s}s")
                except Exception as exc:
                    return DiagnosisStepResult(error=str(exc))

            def _build_extraction_event() -> Dict[str, Any]:
                models_info: Dict[str, Any] = {
                    "extraction": _served(extraction_res.meta, ctx.extraction_spec.foundry_deployment),
                    "reasoning": _reasoning_label(),
                }
                if routing_decision is not None and routing_decision.router_deployment:
                    models_info["router"] = routing_decision.router_deployment
                if extraction_res.error:
                    return {
                        "department": ctx.department_value,
                        "model": ctx.extraction_spec.label,
                        "models": models_info,
                        "error": extraction_res.error,
                    }
                return {
                    "department": ctx.department_value,
                    "patient_story": ctx.patient_story,
                    "routing_signal": ctx.routing_signal,
                    "extracted_features": extraction_res.features,
                    "extraction_sec": extraction_res.extraction_sec,
                    "questions": extraction_res.questions,
                    "question_generation_sec": extraction_res.question_generation_sec,
                    "model": ctx.extraction_spec.label,
                    "models": models_info,
                    "usage": extraction_res.meta,
                }

            def _build_diagnosis_event(res: DiagnosisStepResult, model_label: str) -> Dict[str, Any]:
                if res.error:
                    return {
                        "department": ctx.department_value,
                        "routing_signal": ctx.routing_signal,
                        "model": model_label,
                        "error": res.error,
                        # Additive: the reasoning tier and its fallback both failed.
                        # The note is still delivered with the extraction result.
                        "reasoning_pending": True,
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

            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_s
            pending = {extract_task, diagnose_task}
            extraction_res: Optional[ExtractionStepResult] = None
            diagnosis_res: Optional[DiagnosisStepResult] = None

            while pending:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                done, pending = await asyncio.wait(
                    pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    break
                for finished in done:
                    if finished is extract_task:
                        extraction_res = _extraction_result_from(finished)
                        yield {"event": "extraction", "data": json.dumps(_build_extraction_event())}
                    elif finished is diagnose_task:
                        diagnosis_res = _diagnosis_result_from(finished)
                        yield {
                            "event": "diagnosis",
                            "data": json.dumps(_build_diagnosis_event(diagnosis_res, _reasoning_label())),
                        }

            # Any branch still running when the shared deadline elapsed is
            # cancelled and reported as a timeout error, so both events are
            # always delivered exactly once.
            if extraction_res is None:
                if not extract_task.done():
                    extract_task.cancel()
                extraction_res = ExtractionStepResult(error=f"Timeout after {timeout_s}s")
                yield {"event": "extraction", "data": json.dumps(_build_extraction_event())}
            if diagnosis_res is None:
                if not diagnose_task.done():
                    diagnose_task.cancel()
                diagnosis_res = DiagnosisStepResult(error=f"Timeout after {timeout_s}s")
                yield {
                    "event": "diagnosis",
                    "data": json.dumps(_build_diagnosis_event(diagnosis_res, _reasoning_label())),
                }

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
                # Validation rejected -> one-time escalation to the Sol model.
                # Precedence (owner): skip the escalation if Sol was already
                # attempted for this encounter (served OR failed over to Terra).
                # Bound: one escalation + one fallback hop per reasoning call.
                rejection_msg = "Validation rejected: schema or ICD-10 resolution failure"
                if validation_report.schema_rejection:
                    rejection_msg = f"Schema rejection: {validation_report.schema_rejection.reason}"
                escalation_reason = rejection_msg

                sol_deployment = Config.foundry_model_reasoning_complex()
                terra_deployment = Config.foundry_model_reasoning_complex_fallback()
                served = diagnosis_res.meta.get("served_deployment") if isinstance(diagnosis_res.meta, dict) else None
                sol_already_attempted = (
                    reasoning_spec.foundry_deployment == sol_deployment
                    or served in (sol_deployment, terra_deployment)
                )

                if sol_already_attempted:
                    logger.info(
                        "[Pipeline] Validation failure (%s) but Sol was already attempted "
                        "this encounter (served=%s); skipping escalation.",
                        escalation_reason,
                        served,
                    )
                    yield {
                        "event": "validation_result",
                        "data": json.dumps({
                            "is_valid": validation_passed,
                            "tier": "primary",
                            "escalation_skipped": True,
                            "escalation_reason": escalation_reason,
                            "cautions": validation_dict.get("cautions", []),
                            "missing_fields": validation_dict.get("missing_fields", {}),
                            "validation_latency_ms": validation_dict.get("metrics", {}).get("validation_latency_ms", 0.0),
                            "schema_rejection": validation_dict.get("schema_rejection"),
                        }),
                    }
                else:
                    current_tier = "escalated"

                    logger.warning(
                        "[Pipeline] Validation failure (%s). Escalating reasoning to higher-tier Sol model.",
                        escalation_reason,
                    )

                    sol_spec = get_model("reasoning", escalation=True)
                    escalated_diagnose_task = asyncio.create_task(
                        _run_diagnose(sol_spec, terra_deployment)
                    )
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
                    yield {"event": "diagnosis", "data": json.dumps(_build_diagnosis_event(diagnosis_res, _served(diagnosis_res.meta, sol_spec.label)))}

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
            final_reasoning_model = reasoning_spec.id
            if current_tier == "escalated":
                try:
                    final_reasoning_model = get_model("reasoning", escalation=True).id
                except Exception:
                    final_reasoning_model = "gpt-5.6-sol"

            # Routing audit record (no clinical text) attached to the usage log.
            routing_audit: Optional[Dict[str, Any]] = None
            if routing_decision is not None:
                routing_audit = routing_decision.to_audit_dict()
                diag_meta = getattr(diagnosis_res, "meta", None)
                if isinstance(diag_meta, dict) and diag_meta.get("served_deployment"):
                    routing_audit["served_deployment"] = diag_meta["served_deployment"]
                    routing_audit["fallback_used"] = bool(diag_meta.get("fallback_used"))

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
                routing_audit=routing_audit,
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

