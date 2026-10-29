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
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from llm.runtime import LLMRuntime
from model.llm_service import LLMService
from core.model_registry import ModelSpec
from model.patient_history import check_history_usage, validate_history_citations
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
    routing_decision: Optional[Any] = None
    icd_candidates: Optional[List[Tuple[str, str]]] = None

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
    """Execute clinical feature extraction and question generation."""
    meta: Dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        extracted = await LLMService.extract_features_with_fallback_async(
            translated_text=ctx.patient_story,
            runtime=ctx.runtime,
            department=ctx.department_value,
            is_conversation=ctx.is_conversation,
            model_id=ctx.extraction_spec.id,
            meta=meta,
        )
        raw_features = extracted.get("json_data", {}) if extracted else {}
        features = validate_extraction_output(raw_features, ctx.department_value)
        t_q = time.perf_counter()
        questions, _ = QuestionGenerator.generate(
            extracted_features=features,
            department=ctx.department_value,
        )
        return ExtractionStepResult(
            features=features,
            questions=questions,
            extraction_sec=round(t_q - t0, 3),
            question_generation_sec=round(time.perf_counter() - t_q, 3),
            meta=meta,
        )
    except Exception as exc:
        logger.error("[Pipeline] Extraction step failed: %s", exc)
        return ExtractionStepResult(
            error=str(exc),
            extraction_sec=round(time.perf_counter() - t0, 3),
            meta=meta,
        )


async def step_diagnose(
    ctx: PipelineContext,
    reasoning_spec: Optional[ModelSpec] = None,
    fallback_deployment: Optional[str] = None,
) -> DiagnosisStepResult:
    """Execute ICD-10 differential diagnosis recommendation.

    ``fallback_deployment`` selects the single fallback hop for this call
    (Luna -> gpt-4.1-mini, Sol -> gpt-5.6-terra). When None, the tier's
    configured fallback is inferred from the primary deployment.
    """
    spec = reasoning_spec or ctx.reasoning_spec
    meta: Dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        if ctx.icd_candidates is not None:
            catalogue_candidates = ctx.icd_candidates
        else:
            catalogue_candidates = await asyncio.to_thread(
                get_registry().search_diagnosis_candidates,
                ctx.patient_story,
            )
            ctx.icd_candidates = catalogue_candidates
        diagnosis_result = await LLMService.recommend_diagnosis_with_fallback_async(
            translated_text=ctx.patient_story,
            runtime=ctx.runtime,
            department=ctx.department_value,
            patient_history_text=ctx.patient_history_text,
            is_conversation=ctx.is_conversation,
            model_id=spec.id,
            meta=meta,
            catalogue_candidates=catalogue_candidates,
            fallback_deployment=fallback_deployment,
        )
        raw_data = diagnosis_result.get("json_data", {}) if diagnosis_result else {}
        data = validate_diagnosis_output(raw_data)
        data = constrain_diagnosis_to_registry(
            data,
            ctx.department_value,
            catalogue_candidates,
        )
        if isinstance(data, dict):
            data = validate_history_citations(data, ctx.patient_history_visits)
            if data["history_citation_missing"]:
                logger.warning(
                    "[Pipeline] session_id=%s: claimed history impact lacks a verified visit-date citation",
                    ctx.session_id,
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
