"""
src/validation/middleware.py
============================
FastAPI dependency-injection validation layer for Phase 2.

NOT a traditional ASGI middleware — uses ``Depends()`` for per-route schema
binding so each endpoint opts-in explicitly and the validation path is
transparent in OpenAPI documentation.

Architecture
------------
Input dependencies
  validate_analyze_input      → AnalyzeInput
  validate_refine_field_input → RefineFieldInput
  validate_feedback_input     → FeedbackInput

Output coercion helpers (synchronous, < 0.5 ms)
  validate_extraction_output  → dict (coerced against department schema)
  validate_diagnosis_output   → dict (coerced against DiagnosisOutput)

Error contract
--------------
All input dependencies raise ``HTTPException(422)`` on validation failure,
preserving FastAPI's standard Unprocessable Entity response shape::

    {
        "detail": [
            {"loc": [...], "msg": "...", "type": "..."},
            ...
        ]
    }

Output coercion helpers are *never-fail* — they return the original dict
unchanged on any error, so a validation problem never blocks the SSE stream.

Performance notes
-----------------
* Pydantic v2 model compilation happens once at import time.
* Individual validation calls add < 0.5 ms (see Phase 1 benchmarks).
* Output helpers are called synchronously inside async generators; they
  must stay fast — no I/O, no blocking calls.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Depends, Form, HTTPException
from pydantic import ValidationError

from schemas.input_schemas import AnalyzeInput, FeedbackInput, RefineFieldInput
from schemas.output_schemas import (
    EXTRACTION_SCHEMAS,
    DiagnosisOutput,
    get_extraction_output_schema,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Input validation dependencies
# ══════════════════════════════════════════════════════════════════════════════

async def validate_analyze_input(
    text:               str            = Form(...,   description="Raw transcript from live transcription"),
    language:           str            = Form("en",  description="Audio language: 'en' or 'ar'"),
    mode:               str            = Form("doctor", description="'doctor' or 'conversation'"),
    department:         str            = Form("opd", description="'er', 'opd', or 'dental'"),
    specialty:          Optional[str]  = Form(None,  description="Medical specialty — optional"),
    use_history:        bool           = Form(False, description="Look up patient visit history"),
    patient_id:         Optional[str]  = Form(None,  description="Patient Code — required when use_history is true"),
    business_unit:      Optional[str]  = Form(None,  description="AHJ, AMH or ANC — required when use_history is true"),
    clinician_name:     Optional[str]  = Form(None,  description="Self-reported doctor name"),
    transcription_sec:  Optional[float]= Form(None,  description="Client-measured transcription duration (seconds)"),
    session_id:         Optional[str]  = Form(None,  description="Client-generated session ID"),
    extraction_model:   Optional[str]  = Form(None,  description="Optional doctor-selected extraction model id"),
    reasoning_model:    Optional[str]  = Form(None,  description="Optional doctor-selected reasoning model id"),
) -> AnalyzeInput:
    """FastAPI ``Depends`` — parse and validate ``POST /audio/analyze`` form fields.

    Raises ``HTTPException(422)`` on any validation error so the endpoint
    body never runs with invalid data.  All string fields are Unicode-normalized
    and whitespace-collapsed by the schema's validators.
    """
    # In direct unit tests, parameters might take their default Form(...) objects.
    # We unwrap them to ensure clean values pass into Pydantic.
    def _val(v: Any, default: Any = None) -> Any:
        from fastapi.params import Form as _FormParam
        if isinstance(v, _FormParam):
            return default if v.default is ... else v.default
        return v

    try:
        return AnalyzeInput(
            text=_val(text),
            language=_val(language, "en"),
            mode=_val(mode, "doctor"),
            department=_val(department, "opd"),
            specialty=_val(specialty, None),
            use_history=_val(use_history, False),
            patient_id=_val(patient_id, None),
            business_unit=_val(business_unit, None),
            clinician_name=_val(clinician_name, None),
            transcription_sec=_val(transcription_sec, None),
            session_id=_val(session_id, None),
            extraction_model=_val(extraction_model, None),
            reasoning_model=_val(reasoning_model, None),
        )
    except ValidationError as exc:
        logger.debug("[Middleware] AnalyzeInput validation failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False),
        )


async def validate_refine_field_input(
    text:      str           = Form(...,  description="Raw transcript of the spoken answer for one field"),
    language:  str           = Form("en", description="Language the answer was spoken in: 'en' or 'ar'"),
    specialty: Optional[str] = Form(None, description="Medical specialty — optional"),
) -> RefineFieldInput:
    """FastAPI ``Depends`` — parse and validate ``POST /audio/refine-field`` form fields."""
    def _val(v: Any, default: Any = None) -> Any:
        from fastapi.params import Form as _FormParam
        if isinstance(v, _FormParam):
            return default if v.default is ... else v.default
        return v

    try:
        return RefineFieldInput(
            text=_val(text),
            language=_val(language, "en"),
            specialty=_val(specialty, None),
        )
    except ValidationError as exc:
        logger.debug("[Middleware] RefineFieldInput validation failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False),
        )


async def validate_feedback_input(
    rating:     Optional[str] = Form(None, description='"up", "down", or omitted'),
    text:       str           = Form("",   description="Free-text feedback"),
    name:       Optional[str] = Form(None, description="Self-reported name"),
    department: Optional[str] = Form(None),
    language:   Optional[str] = Form(None),
    session_id: Optional[str] = Form(None, description="Session this feedback is about"),
) -> FeedbackInput:
    """FastAPI ``Depends`` — parse and validate ``POST /feedback`` form fields."""
    def _val(v: Any, default: Any = None) -> Any:
        from fastapi.params import Form as _FormParam
        if isinstance(v, _FormParam):
            return default if v.default is ... else v.default
        return v

    try:
        return FeedbackInput(
            rating=_val(rating, None),
            text=_val(text, ""),
            name=_val(name, None),
            department=_val(department, None),
            language=_val(language, None),
            session_id=_val(session_id, None),
        )
    except ValidationError as exc:
        logger.debug("[Middleware] FeedbackInput validation failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Output validation / coercion helpers
# ══════════════════════════════════════════════════════════════════════════════

def validate_extraction_output(raw: Any, department: str) -> dict[str, Any]:
    """Validate and coerce LLM extraction output against the department schema.

    Behaviour
    ---------
    * On success — returns the model's ``model_dump()`` dict with coerced
      field types (lists → comma-joined strings, ints → strings, etc.).
    * On unknown department — returns ``raw`` unchanged (pass-through).
    * On any Pydantic validation error — logs a warning and returns ``raw``
      unchanged so the SSE stream is never blocked by an output schema issue.

    This is intentionally *never-fail*: output validation is a coercion gate,
    not a hard stop.  The endpoint will still emit the ``extraction`` SSE event
    even if the LLM returned an unexpected shape.
    """
    if not isinstance(raw, dict):
        logger.warning(
            "[Middleware] validate_extraction_output: expected dict, got %s — passing through",
            type(raw).__name__,
        )
        return raw if isinstance(raw, dict) else {}

    dept = (department or "").strip().lower()
    schema_cls = EXTRACTION_SCHEMAS.get(dept)
    if schema_cls is None:
        logger.debug(
            "[Middleware] validate_extraction_output: unknown department %r — passing through",
            department,
        )
        return raw

    try:
        validated = schema_cls.model_validate(raw)
        coerced = validated.model_dump(exclude_none=False)
        # Filter out None values that weren't in the original dict to avoid
        # overwriting genuinely absent fields with explicit nulls in the log.
        return {k: v for k, v in coerced.items() if v is not None or k in raw}
    except Exception as exc:
        logger.warning(
            "[Middleware] validate_extraction_output: schema validation failed for dept=%r — "
            "passing through raw. Error: %s",
            department, exc,
        )
        return raw


def validate_diagnosis_output(raw: Any) -> Any:
    """Validate and coerce LLM diagnosis output against ``DiagnosisOutput``.

    Coerces:
    * ``icd10_codes`` — any to ``List[str]``
    * ``differential_diagnoses`` — any to ``Optional[List[str]]``
    * ``reasoning``, ``history_considered``, ``history_notes`` — any to ``str``
    * ``triage_priority``, ``final_diagnosis`` — any to ``Optional[str]``

    Returns the original ``raw`` dict unchanged on any error.
    """
    if not isinstance(raw, dict):
        logger.warning(
            "[Middleware] validate_diagnosis_output: expected dict, got %s — passing through",
            type(raw).__name__,
        )
        return raw

    try:
        validated = DiagnosisOutput.model_validate(raw)
        coerced = validated.model_dump(exclude_none=False)
        # Preserve None-valued keys only if they were present in the original,
        # to avoid injecting unexpected null fields into the usage log.
        return {k: v for k, v in coerced.items() if v is not None or k in raw}
    except Exception as exc:
        logger.warning(
            "[Middleware] validate_diagnosis_output: schema validation failed — "
            "passing through raw. Error: %s",
            exc,
        )
        return raw
