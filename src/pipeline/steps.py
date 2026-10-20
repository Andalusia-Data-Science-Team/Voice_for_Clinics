"""Pipeline Execution Context and Modular Step Implementations.

Defines PipelineContext dataclass and isolated async callables for:
- step_refine
- step_translate
- step_red_flags
- step_extract
- step_diagnose
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from llm.runtime import LLMRuntime
from model.llm_service import LLMService
from core.model_registry import ModelSpec, get_fallback_extraction_model
from model.patient_history import check_history_usage
from model.question_generator import QuestionGenerator
from model.icd10_registry import get_registry
from model.diagnosis import constrain_diagnosis_to_registry
from schemas.input_schemas import Language, Mode
from utils.red_flags import detect_red_flags
from validation.middleware import (
    validate_diagnosis_output,
    validate_extraction_output,
)
from pipeline.merger import DiagnosisStepResult, ExtractionStepResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Stateless carrier for pipeline execution state and parameters."""
    session_id: str
    text: str
    language: Language
    mode: Mode
    department_value: str
    specialty_value: Optional[str]
    runtime: LLMRuntime
    extraction_spec: ModelSpec
    reasoning_spec: ModelSpec
    
    patient_id: Optional[str] = None
    business_unit_value: Optional[str] = None
    use_history: bool = False
    patient_history_visits: List[Any] = field(default_factory=list)
    patient_history_text: str = ""
    history_error: Optional[str] = None
    transcription_sec: Optional[float] = None
    clinician_name: Optional[str] = None
    
    # State populated during pipeline steps
    pipeline_start: float = field(default_factory=time.perf_counter)
    refined_text: str = ""
    refinement_sec: float = 0.0
    refinement_skipped: bool = False
    translated_text: str = ""
    translation_sec: float = 0.0
    patient_story: str = ""
    routing_signal: Optional[Dict[str, Any]] = None

    @property
    def is_arabic(self) -> bool:
        return self.language == Language.arabic

    @property
    def is_conversation(self) -> bool:
        return self.mode == Mode.conversation

    @property
    def skip_refinement(self) -> bool:
        return self.mode == Mode.doctor


async def step_refine_stream(
    ctx: PipelineContext,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute transcription refinement step, yielding SSE event dicts."""
    t0 = time.perf_counter()
    if ctx.skip_refinement:
        ctx.refined_text = ctx.text
        ctx.refinement_sec = 0.0
        ctx.refinement_skipped = True
        yield {
            "event": "refinement_skipped",
            "data": {"refined_text": ctx.refined_text},
        }
        return

    refined_parts: List[str] = []
    try:
        if ctx.is_arabic:
            stream = LLMService.refine_ar_transcription_stream_async(
                ctx.text,
                ctx.runtime,
                is_conversation=ctx.is_conversation,
                specialty=ctx.specialty_value,
            )
        else:
            stream = LLMService.refine_en_transcription_stream_async(
                ctx.text,
                ctx.runtime,
                is_conversation=ctx.is_conversation,
                specialty=ctx.specialty_value,
            )
        async for delta in stream:
            refined_parts.append(delta)
            yield {"event": "refinement_delta", "data": {"delta": delta}}
    except Exception as exc:
        logger.error("[Pipeline] Refinement failed: %s", exc)
        yield {"event": "error", "data": {"step": "refinement", "detail": str(exc)}}
        return

    ctx.refined_text = "".join(refined_parts)
    ctx.refinement_sec = round(time.perf_counter() - t0, 3)
    yield {
        "event": "refinement_done",
        "data": {
            "refined_text": ctx.refined_text,
            "refinement_sec": ctx.refinement_sec,
        },
    }


async def step_translate_stream(
    ctx: PipelineContext,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute Arabic translation step if applicable, yielding SSE event dicts."""
    if ctx.is_arabic:
        t0 = time.perf_counter()
        translated_parts: List[str] = []
        try:
            stream = LLMService.translate_to_eng_stream_async(
                ctx.refined_text,
                ctx.runtime,
                is_conversation=ctx.is_conversation,
            )
            async for delta in stream:
                translated_parts.append(delta)
                yield {"event": "translation_delta", "data": {"delta": delta}}
        except Exception as exc:
            logger.error("[Pipeline] Translation failed: %s", exc)
            yield {"event": "error", "data": {"step": "translation", "detail": str(exc)}}
            return

        ctx.translated_text = "".join(translated_parts)
        ctx.translation_sec = round(time.perf_counter() - t0, 3)
        yield {
            "event": "translation_done",
            "data": {
                "final_text": ctx.translated_text,
                "translation_sec": ctx.translation_sec,
            },
        }
    else:
        ctx.translated_text = ctx.refined_text
        ctx.translation_sec = 0.0


def step_red_flags(ctx: PipelineContext) -> Dict[str, Any]:
    """Detect red flag routing signals from text progression."""
    signal = detect_red_flags("\n".join((ctx.text, ctx.refined_text, ctx.patient_story)))
    ctx.routing_signal = signal
    return signal


async def step_extract(ctx: PipelineContext) -> ExtractionStepResult:
    """Execute clinical feature extraction with automatic fallback to gpt-4.1-mini.

    Primary model: ctx.extraction_spec (e.g. gpt-5.6-luna)
    Fallback model: FOUNDRY_FALLBACK_EXTRACTION_MODEL (gpt-4.1-mini) — only used
    when the primary fails after all retries, never shown in UI.
    """
    meta: Dict[str, Any] = {}
    t0 = time.perf_counter()

    async def _attempt(spec: ModelSpec, is_fallback: bool = False) -> ExtractionStepResult:
        """Run one extraction attempt with the given spec."""
        try:
            extracted = await LLMService.extract_features_async(
                translated_text=ctx.patient_story,
                runtime=ctx.runtime,
                department=ctx.department_value,
                is_conversation=ctx.is_conversation,
                model_id=spec.id,
                meta=meta,
            )
            raw_features = extracted.get("json_data", {}) if extracted else {}
            features = validate_extraction_output(raw_features, ctx.department_value)
            t_q = time.perf_counter()
            questions, _ = QuestionGenerator.generate(
                extracted_features=features,
                department=ctx.department_value,
            )
            if is_fallback:
                logger.info(
                    "[Pipeline] Extraction fallback succeeded with model=%s", spec.id
                )
            return ExtractionStepResult(
                features=features,
                questions=questions,
                extraction_sec=round(t_q - t0, 3),
                question_generation_sec=round(time.perf_counter() - t_q, 3),
                meta=meta,
                fallback_model_used=spec.id if is_fallback else None,
            )
        except Exception as exc:
            return None, exc  # type: ignore[return-value]

    # ── Primary attempt ────────────────────────────────────────────────────
    primary_result = await _attempt(ctx.extraction_spec, is_fallback=False)
    if primary_result is not None and not isinstance(primary_result, tuple):
        return primary_result

    primary_error = primary_result[1] if isinstance(primary_result, tuple) else Exception("Unknown")
    logger.warning(
        "[Pipeline] Primary extraction model=%s failed (%s). Trying fallback.",
        ctx.extraction_spec.id, primary_error,
    )

    # ── Fallback attempt (gpt-4.1-mini) ────────────────────────────────────
    try:
        fallback_spec = get_fallback_extraction_model()
        if fallback_spec.id == ctx.extraction_spec.id:
            # Primary IS the fallback model — don't retry the same thing
            raise ValueError("Primary and fallback extraction model are the same")

        fallback_result = await _attempt(fallback_spec, is_fallback=True)
        if fallback_result is not None and not isinstance(fallback_result, tuple):
            return fallback_result

        fallback_error = fallback_result[1] if isinstance(fallback_result, tuple) else Exception("Unknown")
        logger.error(
            "[Pipeline] Fallback extraction model=%s also failed: %s",
            fallback_spec.id, fallback_error,
        )
        return ExtractionStepResult(
            error=f"Primary ({ctx.extraction_spec.id}): {primary_error}; Fallback ({fallback_spec.id}): {fallback_error}",
            extraction_sec=round(time.perf_counter() - t0, 3),
            meta=meta,
        )
    except Exception as exc:
        logger.error("[Pipeline] Extraction fallback mechanism failed: %s", exc)
        return ExtractionStepResult(
            error=str(primary_error),
            extraction_sec=round(time.perf_counter() - t0, 3),
            meta=meta,
        )


async def step_diagnose(
    ctx: PipelineContext,
    reasoning_spec: Optional[ModelSpec] = None,
) -> DiagnosisStepResult:
    """Execute ICD-10 differential diagnosis recommendation."""
    spec = reasoning_spec or ctx.reasoning_spec
    meta: Dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        catalogue_candidates = get_registry().search_diagnosis_candidates(
            ctx.patient_story
        )
        diagnosis_result = await LLMService.recommend_diagnosis_async(
            translated_text=ctx.patient_story,
            runtime=ctx.runtime,
            department=ctx.department_value,
            patient_history_text=ctx.patient_history_text,
            is_conversation=ctx.is_conversation,
            model_id=spec.id,
            meta=meta,
            catalogue_candidates=catalogue_candidates,
        )
        raw_data = diagnosis_result.get("json_data", {}) if diagnosis_result else {}
        data = validate_diagnosis_output(raw_data)
        data = constrain_diagnosis_to_registry(
            data,
            ctx.department_value,
            catalogue_candidates,
        )
        
        history_check = None
        if ctx.patient_history_text and data:
            history_check = check_history_usage(ctx.patient_history_visits, data)
            if history_check["flag"] == "claims_used_no_overlap":
                logger.warning(
                    "[Pipeline] patient_code=%s: history_considered=Yes but no ICD-10 family overlap — history_notes=%r",
                    ctx.patient_id,
                    data.get("history_notes"),
                )
            elif history_check["flag"] == "overlap_but_not_flagged":
                logger.warning(
                    "[Pipeline] patient_code=%s: ICD-10 family overlap %s but history_considered was not 'Yes'",
                    ctx.patient_id,
                    history_check["overlap_families"],
                )

        return DiagnosisStepResult(
            diagnosis=data,
            diagnosis_sec=round(time.perf_counter() - t0, 3),
            meta=meta,
            history_check=history_check,
        )
    except Exception as exc:
        logger.error("[Pipeline] Diagnosis step failed: %s", exc)
        return DiagnosisStepResult(
            error=str(exc),
            diagnosis_sec=round(time.perf_counter() - t0, 3),
            meta=meta,
        )
