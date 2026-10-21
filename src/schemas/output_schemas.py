"""
src/schemas/output_schemas.py
=============================
Strict Pydantic v2 output schemas for every structured result produced by
the MedVoice pipeline.

Architecture notes
------------------
1.  **Department extraction schemas are keyed to ``department_fields.py``.**
    Every field key in ``DEPARTMENT_FIELDS`` has a corresponding Optional[str]
    attribute here, ensuring the extraction-to-schema bridge never silently
    drops or misnames a field.

2.  **Coercion over rejection for output schemas.**  LLM output is inherently
    noisy — models sometimes return ints, lists, or dicts where a string is
    expected.  The ``_coerce_str`` validator converts all these shapes to clean
    strings rather than raising.  This mirrors the strategy already in
    ``validation/schemas.py`` so both layers stay consistent.

3.  **``Optional[str]`` (not ``str = ""``) for extraction fields.**  The
    plan spec uses Optional so that the frontend can distinguish between
    "field was not mentioned" (``None``) and "field was mentioned as empty"
    (``""``).  The ``coerce_to_str`` helper in ``validators.py`` always
    produces a clean ``str``; ``None`` is preserved only when the LLM truly
    omitted the key.

4.  **``EXTRACTION_SCHEMAS`` registry** maps department strings to schema
    classes.  Code that currently does ``dict[department]`` lookups will
    migrate to this registry in Phase 2.

5.  **``PipelineSessionOutput``** is the validated superset of everything
    written to the usage log / MongoDB.  It is not returned directly to the
    client (the SSE stream sends individual event payloads) but serves as the
    authoritative shape contract for the log writer and dashboards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.validators import coerce_to_str, coerce_to_str_list


# ══════════════════════════════════════════════════════════════════════════════
# Shared base config for all output schemas
# ══════════════════════════════════════════════════════════════════════════════

class _OutputBase(BaseModel):
    """Base config for all pipeline output models.

    * ``extra="ignore"`` — extra LLM keys are silently dropped; we never
      want a validation error just because the model invented a field.
    * ``str_strip_whitespace=True`` — automatic edge-whitespace trim.
    * ``validate_assignment=True`` — safe for test mutation.
    * ``populate_by_name=True`` — alias support where needed.
    """
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        validate_assignment=True,
        populate_by_name=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Shared sub-models
# ══════════════════════════════════════════════════════════════════════════════

class QuestionItem(_OutputBase):
    """A single entry in the 'Questions for the doctor' list.

    Generated deterministically from missing extraction fields by
    ``QuestionGenerator``.
    """
    question: str = Field(..., description="The question to ask the clinician")
    answer: Optional[str] = Field(None, description="Answer filled in by the clinician (if any)")
    needs_asking: bool = Field(True, description="False once the field is filled in")
    category: str = Field(..., description="Clinical category (e.g. 'History', 'Examination')")


class DiagnosisOutput(_OutputBase):
    """Structured output of the ICD-10 recommendation step (``run_diagnosis``).

    ER-specific fields (``differential_diagnoses``, ``triage_priority``) are
    always present in the schema but will be ``None`` / ``[]`` for OPD / Dental
    sessions.  The extraction pipeline strips them for non-ER departments before
    persisting; having them here lets the schema validate ER outputs without a
    separate class.
    """
    icd10_codes: List[str] = Field(
        default_factory=list,
        description="Recommended ICD-10 codes, most-likely first",
    )
    icd10_unmatched: List[str] = Field(
        default_factory=list,
        description="Model suggestions excluded because they did not match the ICD-10-AM catalogue",
    )
    reasoning: str = Field("", description="Step-by-step clinical reasoning")
    final_diagnosis: Optional[str] = Field(
        None,
        description="Free-text final diagnosis summary (may be absent if LLM omits it)",
    )
    history_considered: str = Field(
        "No",
        description="Whether the patient's history was factored in ('Yes' / 'No')",
    )
    history_notes: str = Field(
        "",
        description="LLM note on how history influenced the recommendation",
    )
    # ER-only ─────────────────────────────────────────────────────────────────
    differential_diagnoses: Optional[List[str]] = Field(
        None,
        description="ER-only: ranked differential diagnoses",
    )
    differential_unmatched: Optional[List[str]] = Field(
        None,
        description="ER-only: differential suggestions excluded by the ICD-10-AM catalogue",
    )
    triage_priority: Optional[str] = Field(
        None,
        description="ER-only: triage category (e.g. 'Category 2 – Emergency')",
    )

    @field_validator("icd10_codes", "icd10_unmatched", mode="before")
    @classmethod
    def _coerce_icd10_list(cls, v: Any) -> List[str]:
        """Code lists use [] instead of None."""
        return coerce_to_str_list(v)

    @field_validator(
        "differential_diagnoses",
        "differential_unmatched",
        mode="before",
    )
    @classmethod
    def _coerce_differential_list(cls, v: Any) -> Optional[List[str]]:
        """differential_diagnoses is Optional[List[str]] — None passes through."""
        if v is None:
            return None
        return coerce_to_str_list(v)

    @field_validator("reasoning", "history_considered", "history_notes", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        return coerce_to_str(v)

    @field_validator("final_diagnosis", "triage_priority", mode="before")
    @classmethod
    def _coerce_opt_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = coerce_to_str(v)
        return cleaned if cleaned else None


class UsageMeta(_OutputBase):
    """Token / cost / latency metadata for one LLM call.

    Populated by ``_record_meta()`` in ``model/llm_service.py`` and attached
    to both the extraction and diagnosis results in the usage log.
    """
    model_id: Optional[str] = Field(None, description="Model identifier (from ModelSpec.id)")
    foundry_deployment: Optional[str] = Field(None, description="Azure AI Foundry deployment name")
    latency_sec: Optional[float] = Field(None, ge=0.0, description="End-to-end LLM call latency")
    prompt_tokens: Optional[int] = Field(None, ge=0)
    completion_tokens: Optional[int] = Field(None, ge=0)
    reasoning_tokens: Optional[int] = Field(None, ge=0, description="Thinking/reasoning tokens (o1-family only)")
    cost_usd: Optional[float] = Field(None, ge=0.0)
    reasoning_enabled: Optional[bool] = Field(None, description="Whether extended reasoning was active")
    error: Optional[str] = Field(None, description="Populated when the LLM call failed parsing")
    served_deployment: Optional[str] = Field(
        None,
        description="Deployment that actually served the call (primary or fallback)",
    )
    fallback_used: Optional[bool] = Field(
        None,
        description="True when a single fallback hop served the call",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Per-department extraction schemas
# ══════════════════════════════════════════════════════════════════════════════
# Field keys mirror DEPARTMENT_FIELDS in model/department_fields.py exactly.
# Optional[str] = None means the LLM omitted the field entirely; empty string
# means it was mentioned but left blank.

class _ExtractionBase(_OutputBase):
    """Coercion base for all extraction schemas."""

    @classmethod
    def _coerce_field(cls, v: Any) -> Optional[str]:
        """Convert LLM output to Optional[str]: None if absent, str otherwise."""
        if v is None:
            return None
        cleaned = coerce_to_str(v)
        return cleaned if cleaned else None


# ── ER ────────────────────────────────────────────────────────────────────────

class ERExtractionOutput(_ExtractionBase):
    """Emergency Room clinical-note extraction output.

    Field order matches ``DEPARTMENT_FIELDS["er"]`` in department_fields.py.
    """
    chief_complaint:        Optional[str] = None
    onset:                  Optional[str] = None
    course:                 Optional[str] = None
    duration:               Optional[str] = None
    past_medical_history:   Optional[str] = None
    past_surgical_history:  Optional[str] = None
    medication_history:     Optional[str] = None
    allergy_status:         Optional[str] = None
    family_history:         Optional[str] = None
    social_history:         Optional[str] = None
    airway:                 Optional[str] = None
    breathing:              Optional[str] = None
    circulation:            Optional[str] = None
    disability:             Optional[str] = None
    exposure:               Optional[str] = None
    examination:            Optional[str] = None
    investigations:         Optional[str] = None
    plan_of_care:           Optional[str] = None
    requested_consultation: Optional[str] = None
    discharge_instructions: Optional[str] = None
    patient_disposition:    Optional[str] = None
    sick_leave:             Optional[str] = None

    @field_validator(
        "chief_complaint", "onset", "course", "duration",
        "past_medical_history", "past_surgical_history", "medication_history",
        "allergy_status", "family_history", "social_history",
        "airway", "breathing", "circulation", "disability", "exposure",
        "examination", "investigations", "plan_of_care",
        "requested_consultation", "discharge_instructions",
        "patient_disposition", "sick_leave",
        mode="before",
    )
    @classmethod
    def _coerce(cls, v: Any) -> Optional[str]:
        return cls._coerce_field(v)


# ── OPD ──────────────────────────────────────────────────────────────────────

class OPDExtractionOutput(_ExtractionBase):
    """Outpatient Department clinical-note extraction output.

    Field order matches ``DEPARTMENT_FIELDS["opd"]`` in department_fields.py.
    """
    allergy_status:          Optional[str] = None
    chief_complaint:         Optional[str] = None
    onset:                   Optional[str] = None
    course:                  Optional[str] = None
    duration:                Optional[str] = None
    comorbidities:           Optional[str] = None
    medication_history:      Optional[str] = None
    past_medical_history:    Optional[str] = None
    past_surgical_history:   Optional[str] = None
    social_family_history:   Optional[str] = None
    examination_finding:     Optional[str] = None
    plan_of_care:            Optional[str] = None
    procedures_planned:      Optional[str] = None
    referrals:               Optional[str] = None
    follow_up_timing:        Optional[str] = None
    patient_instructions:    Optional[str] = None
    outcome_progress_status: Optional[str] = None

    @field_validator(
        "allergy_status", "chief_complaint", "onset", "course", "duration",
        "comorbidities", "medication_history", "past_medical_history",
        "past_surgical_history", "social_family_history", "examination_finding",
        "plan_of_care", "procedures_planned", "referrals", "follow_up_timing",
        "patient_instructions", "outcome_progress_status",
        mode="before",
    )
    @classmethod
    def _coerce(cls, v: Any) -> Optional[str]:
        return cls._coerce_field(v)


# ── Dental ────────────────────────────────────────────────────────────────────

class DentalExtractionOutput(_ExtractionBase):
    """Dental encounter clinical-note extraction output.

    Field order matches ``DEPARTMENT_FIELDS["dental"]`` in department_fields.py.
    """
    chief_complaint:         Optional[str] = None
    onset:                   Optional[str] = None
    course:                  Optional[str] = None
    duration:                Optional[str] = None
    past_medical_history:    Optional[str] = None
    consent_obtained:        Optional[str] = None
    treatment_plan_status:   Optional[str] = None
    charge_category:         Optional[str] = None
    patient_education_given: Optional[str] = None
    radiographs_taken:       Optional[str] = None
    tooth_number:            Optional[str] = None
    subspecialty_focus:      Optional[str] = None
    procedure_executed:      Optional[str] = None
    materials_inventory:     Optional[str] = None
    re_make:                 Optional[str] = None
    complication_if_present: Optional[str] = None
    internal_referral:       Optional[str] = None
    service_leakage_alert:   Optional[str] = None
    next_visit_tooth_number: Optional[str] = None
    next_visit_treatment:    Optional[str] = None
    retention_priority:      Optional[str] = None
    followup_timeline:       Optional[str] = None

    @field_validator(
        "chief_complaint", "onset", "course", "duration",
        "past_medical_history", "consent_obtained", "treatment_plan_status",
        "charge_category", "patient_education_given", "radiographs_taken",
        "tooth_number", "subspecialty_focus", "procedure_executed",
        "materials_inventory", "re_make", "complication_if_present",
        "internal_referral", "service_leakage_alert",
        "next_visit_tooth_number", "next_visit_treatment",
        "retention_priority", "followup_timeline",
        mode="before",
    )
    @classmethod
    def _coerce(cls, v: Any) -> Optional[str]:
        return cls._coerce_field(v)


# ══════════════════════════════════════════════════════════════════════════════
# Department → schema registry
# ══════════════════════════════════════════════════════════════════════════════

#: Maps department string → extraction schema class.
#: Mirrors ``DEPARTMENT_SCHEMA_MAP`` in ``validation/schemas.py`` but uses the
#: new Optional-field schemas from this module.
EXTRACTION_SCHEMAS: Dict[str, Type[_ExtractionBase]] = {
    "er":     ERExtractionOutput,
    "opd":    OPDExtractionOutput,
    "dental": DentalExtractionOutput,
}


def get_extraction_output_schema(department: str) -> Type[_ExtractionBase]:
    """Return the extraction output schema class for ``department``.

    Falls back to ``OPDExtractionOutput`` for unknown departments, matching the
    existing fallback in ``model/department_fields.py:get_department_fields()``.
    """
    return EXTRACTION_SCHEMAS.get((department or "").strip().lower(), OPDExtractionOutput)


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline session output (usage-log / MongoDB envelope)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineSessionOutput(_OutputBase):
    """Complete validated output of one ``/audio/analyze`` session.

    This is the authoritative contract for everything that gets written to the
    usage log (``src/logs/usage.jsonl``) and, eventually, MongoDB.  It is NOT
    returned directly to the client — the SSE stream sends individual event
    payloads — but it validates the final ``log`` dict before persistence.

    Timings are all in seconds and are ``None`` when the corresponding step was
    skipped (e.g. ``translation_sec=None`` for English sessions).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    session_id: Optional[str] = None
    status: str = Field("ok", description="'ok' or 'error'")
    department: str = Field(..., description="er | opd | dental")
    language: str = Field(..., description="ar | en")
    mode: str = Field(..., description="doctor | conversation")
    specialty: Optional[str] = None

    # ── Timings (seconds) ─────────────────────────────────────────────────────
    transcription_sec:        Optional[float] = Field(None, ge=0.0)
    refinement_sec:           Optional[float] = Field(None, ge=0.0)
    translation_sec:          Optional[float] = Field(None, ge=0.0)
    extraction_sec:           Optional[float] = Field(None, ge=0.0)
    question_generation_sec:  Optional[float] = Field(None, ge=0.0)
    diagnosis_sec:            Optional[float] = Field(None, ge=0.0)
    total_sec:                Optional[float] = Field(None, ge=0.0)

    # ── Text artifacts ─────────────────────────────────────────────────────────
    transcript:      Optional[str] = None
    refined_text:    Optional[str] = None
    translated_text: Optional[str] = None
    patient_story:   Optional[str] = None

    # ── Structured outputs ─────────────────────────────────────────────────────
    extracted_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw extraction dict from LLM (not yet validated against department schema)",
    )
    routing_signal: Optional[Dict[str, Any]] = Field(
        None,
        description="Lexicon-based candidate red-flag matches for clinician review",
    )
    routing_audit: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Routing decision audit record (complexity, reason_codes, tier, "
            "serving deployment, timings). Never contains clinical text."
        ),
    )
    questions: Optional[List[QuestionItem]] = Field(
        None,
        description="Questions for the doctor, derived deterministically from missing fields",
    )
    diagnosis: Optional[DiagnosisOutput] = None

    # ── LLM usage metadata ─────────────────────────────────────────────────────
    extraction_usage: Optional[UsageMeta] = Field(
        None,
        description="Token / latency / cost info for the extraction LLM call",
    )
    diagnosis_usage: Optional[UsageMeta] = Field(
        None,
        description="Token / latency / cost info for the diagnosis LLM call",
    )
