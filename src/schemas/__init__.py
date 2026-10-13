"""
src/schemas/__init__.py
=======================
Public re-exports for the ``schemas`` package.

Importing from here (``from schemas import AnalyzeInput``) rather than from
the sub-modules directly keeps downstream code decoupled from the internal
file layout and gives us a single place to control what is considered public
API.

Input schemas
-------------
    AnalyzeInput         — POST /audio/analyze
    RefineFieldInput     — POST /audio/refine-field
    WebSocketInitInput   — WS  /ws/transcribe (init message)
    FeedbackInput        — POST /feedback

Enums
-----
    Language, Mode, Department, BusinessUnit

Output schemas
--------------
    ERExtractionOutput     — ER clinical-note extraction result
    OPDExtractionOutput    — OPD clinical-note extraction result
    DentalExtractionOutput — Dental clinical-note extraction result
    DiagnosisOutput        — ICD-10 recommendation result
    QuestionItem           — Single missing-field question
    UsageMeta              — LLM call token/cost/latency metadata
    PipelineSessionOutput  — Full session envelope (usage log / MongoDB shape)

Registry helper
---------------
    EXTRACTION_SCHEMAS          — dict[str, Type[_ExtractionBase]]
    get_extraction_output_schema — department → schema class lookup
"""

from schemas.input_schemas import (
    AnalyzeInput,
    BusinessUnit,
    Department,
    FeedbackInput,
    Language,
    Mode,
    RefineFieldInput,
    WebSocketInitInput,
)
from schemas.output_schemas import (
    DentalExtractionOutput,
    DiagnosisOutput,
    ERExtractionOutput,
    EXTRACTION_SCHEMAS,
    OPDExtractionOutput,
    PipelineSessionOutput,
    QuestionItem,
    UsageMeta,
    get_extraction_output_schema,
)
from schemas.validators import (
    MAX_FIELD_TEXT_LEN,
    MAX_NAME_LEN,
    MAX_PATIENT_ID_LEN,
    MAX_SESSION_ID_LEN,
    MAX_TEXT_LEN,
    clamp_float,
    coerce_to_str,
    coerce_to_str_list,
    normalize_patient_id,
    require_non_blank,
    sanitize_medical_text,
)

__all__ = [
    # ── Input schemas ──────────────────────────────────────────────────────
    "AnalyzeInput",
    "RefineFieldInput",
    "WebSocketInitInput",
    "FeedbackInput",
    # ── Enums ──────────────────────────────────────────────────────────────
    "Language",
    "Mode",
    "Department",
    "BusinessUnit",
    # ── Output schemas ─────────────────────────────────────────────────────
    "ERExtractionOutput",
    "OPDExtractionOutput",
    "DentalExtractionOutput",
    "DiagnosisOutput",
    "QuestionItem",
    "UsageMeta",
    "PipelineSessionOutput",
    # ── Registry ──────────────────────────────────────────────────────────
    "EXTRACTION_SCHEMAS",
    "get_extraction_output_schema",
    # ── Validators / helpers ───────────────────────────────────────────────
    "sanitize_medical_text",
    "coerce_to_str",
    "coerce_to_str_list",
    "normalize_patient_id",
    "clamp_float",
    "require_non_blank",
    # ── Constants ──────────────────────────────────────────────────────────
    "MAX_TEXT_LEN",
    "MAX_FIELD_TEXT_LEN",
    "MAX_PATIENT_ID_LEN",
    "MAX_SESSION_ID_LEN",
    "MAX_NAME_LEN",
]
