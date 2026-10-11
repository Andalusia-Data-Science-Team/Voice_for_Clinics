"""
MedVoice Validation Subsystem - Strict Pydantic Clinical Schemas
================================================================
Part of Phase 2 (Milestone 3.2): Strict Pydantic Output Schemas & Validation Contracts.

Provides typed, robust, high-performance Pydantic v2 schemas:
  - ClinicalExtractionBase: Common base for all clinical extraction shapes
  - ERExtractionSchema: Strict validation for Emergency Department (Primary survey, etc.)
  - OPDExtractionSchema: Outpatient Department clinical notes
  - DentalExtractionSchema: Dental charting, procedures, materials, and teeth numbers
  - DiagnosisReasoningSchema: ICD-10 codes, reasoning, differential diagnoses, triage
  - Safe parsing & Rejection Engine helper methods
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from validation.exceptions import (
    ClinicalSchemaValidationError,
    FieldValidationError,
    SchemaRejectionPayload,
    ValidationErrorCode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Schema with fast coercion and whitespace cleaning
# ---------------------------------------------------------------------------

class ClinicalDataBaseModel(BaseModel):
    """Base Pydantic v2 model for all clinical contracts."""
    model_config = ConfigDict(
        extra="ignore",             # Ignore unexpected extra attributes safely
        str_strip_whitespace=True,  # Automatically trim strings
        validate_assignment=True,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Extraction Schemas (Department-specific)
# ---------------------------------------------------------------------------

class ClinicalExtractionBase(ClinicalDataBaseModel):
    """Core clinical extraction fields common across specialties."""
    chief_complaint: str = Field(default="", description="Primary complaint or reason for visit")
    onset: str = Field(default="", description="When/how the complaint started")
    course: str = Field(default="", description="Progression over time")
    duration: str = Field(default="", description="Duration of symptoms")
    past_medical_history: str = Field(default="", description="Relevant past medical history")
    past_surgical_history: str = Field(default="", description="Past surgical history")
    medication_history: str = Field(default="", description="Current home medications")
    allergy_status: str = Field(default="", description="Documented drug/food allergies or NKDA")

    @field_validator(
        "chief_complaint", "onset", "course", "duration",
        "past_medical_history", "past_surgical_history",
        "medication_history", "allergy_status",
        mode="before",
    )
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(item).strip() for item in v if item is not None)
        return str(v).strip()


class ERExtractionSchema(ClinicalExtractionBase):
    """Emergency Room (ER) clinical encounter schema."""
    family_history: str = Field(default="")
    social_history: str = Field(default="")
    airway: str = Field(default="", description="Primary survey: Airway (A)")
    breathing: str = Field(default="", description="Primary survey: Breathing (B)")
    circulation: str = Field(default="", description="Primary survey: Circulation (C)")
    disability: str = Field(default="", description="Primary survey: Disability/Neuro (D)")
    exposure: str = Field(default="", description="Primary survey: Exposure (E)")
    examination: str = Field(default="", description="General/focused physical exam")
    investigations: str = Field(default="", description="Labs, ECG, imaging results")
    plan_of_care: str = Field(default="", description="Emergency management plan")
    requested_consultation: str = Field(default="")
    discharge_instructions: str = Field(default="")
    patient_disposition: str = Field(default="")
    sick_leave: str = Field(default="")

    @field_validator(
        "family_history", "social_history", "airway", "breathing",
        "circulation", "disability", "exposure", "examination",
        "investigations", "plan_of_care", "requested_consultation",
        "discharge_instructions", "patient_disposition", "sick_leave",
        mode="before",
    )
    @classmethod
    def _coerce_er_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(item).strip() for item in v if item is not None)
        return str(v).strip()


class OPDExtractionSchema(ClinicalExtractionBase):
    """Outpatient Department (OPD) clinical visit schema."""
    social_family_history: str = Field(default="")
    comorbidities: str = Field(default="")
    examination_finding: str = Field(default="")
    plan_of_care: str = Field(default="")
    procedures_planned: str = Field(default="")
    referrals: str = Field(default="")
    follow_up_timing: str = Field(default="")
    patient_instructions: str = Field(default="")
    outcome_progress_status: str = Field(default="")

    @field_validator(
        "social_family_history", "comorbidities", "examination_finding",
        "plan_of_care", "procedures_planned", "referrals",
        "follow_up_timing", "patient_instructions", "outcome_progress_status",
        mode="before",
    )
    @classmethod
    def _coerce_opd_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(item).strip() for item in v if item is not None)
        return str(v).strip()


class DentalExtractionSchema(ClinicalExtractionBase):
    """Dental encounter clinical schema."""
    consent_obtained: str = Field(default="")
    treatment_plan_status: str = Field(default="")
    charge_category: str = Field(default="")
    patient_education_given: str = Field(default="")
    radiographs_taken: str = Field(default="")
    tooth_number: str = Field(default="")
    subspecialty_focus: str = Field(default="")
    procedure_executed: str = Field(default="")
    materials_inventory: str = Field(default="")
    re_make: str = Field(default="")
    complication_if_present: str = Field(default="")
    internal_referral: str = Field(default="")
    service_leakage_alert: str = Field(default="")
    next_visit_tooth_number: str = Field(default="")
    next_visit_treatment: str = Field(default="")
    retention_priority: str = Field(default="")
    followup_timeline: str = Field(default="")

    @field_validator(
        "consent_obtained", "treatment_plan_status", "charge_category",
        "patient_education_given", "radiographs_taken", "tooth_number",
        "subspecialty_focus", "procedure_executed", "materials_inventory",
        "re_make", "complication_if_present", "internal_referral",
        "service_leakage_alert", "next_visit_tooth_number",
        "next_visit_treatment", "retention_priority", "followup_timeline",
        mode="before",
    )
    @classmethod
    def _coerce_dental_str(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ", ".join(str(item).strip() for item in v if item is not None)
        return str(v).strip()


# ---------------------------------------------------------------------------
# Diagnosis & Reasoning Schema
# ---------------------------------------------------------------------------

class DiagnosisReasoningSchema(ClinicalDataBaseModel):
    """Clinical diagnosis recommendation and reasoning schema."""
    icd10_codes: List[str] = Field(default_factory=list, description="Recommended ICD-10 codes")
    reasoning: str = Field(default="", description="Clinical reasoning explanation")
    history_considered: Optional[str] = Field(default=None, description="Whether past visits were factored in ('Yes'/'No')")
    history_notes: Optional[str] = Field(default="", description="Notes on how history was incorporated")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Alternative differential diagnoses (ER)")
    triage_priority: Optional[str] = Field(default=None, description="Emergency triage category (ER)")

    @field_validator("icd10_codes", "differential_diagnoses", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if isinstance(v, dict):
            v = list(v.values())
        return [str(item).strip() for item in v if item is not None and str(item).strip()]

    @field_validator("reasoning", "history_notes", mode="before")
    @classmethod
    def _coerce_opt_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


# ---------------------------------------------------------------------------
# Department Registry Mapping
# ---------------------------------------------------------------------------

DEPARTMENT_SCHEMA_MAP: Dict[str, Type[ClinicalExtractionBase]] = {
    "er": ERExtractionSchema,
    "opd": OPDExtractionSchema,
    "dental": DentalExtractionSchema,
}


def get_extraction_schema_for_department(department: str) -> Type[ClinicalExtractionBase]:
    """Retrieve the corresponding Pydantic schema class for a department."""
    dept = (department or "").strip().lower()
    return DEPARTMENT_SCHEMA_MAP.get(dept, OPDExtractionSchema)


# ---------------------------------------------------------------------------
# Rejection & Parsing Engine
# ---------------------------------------------------------------------------

class SchemaValidator:
    """High-performance validator with graceful error trapping and rejection reasons."""

    @staticmethod
    def validate_extraction(
        data: Any,
        department: str = "opd",
    ) -> Tuple[Optional[ClinicalExtractionBase], Optional[SchemaRejectionPayload]]:
        """Validate clinical extraction output against department schema.

        Returns:
            (validated_instance, None) on success
            (None, rejection_payload) on validation error
        """
        if not isinstance(data, dict):
            rejection = SchemaRejectionPayload(
                status="rejected",
                error_code=ValidationErrorCode.UNEXPECTED_DATA_SHAPE.value,
                reason=f"Extraction data must be a dictionary, received {type(data).__name__}",
                loc=["root"],
                errors=[{"loc": ["root"], "reason": f"Expected dict, got {type(data).__name__}", "error_code": "type_error"}],
            )
            return None, rejection

        schema_cls = get_extraction_schema_for_department(department)
        try:
            validated = schema_cls.model_validate(data)
            return validated, None
        except ValidationError as ve:
            field_errors = []
            locs = []
            for err in ve.errors():
                loc_tuple = tuple(str(x) for x in err.get("loc", []))
                loc_str = ".".join(loc_tuple)
                locs.append(loc_str)
                msg = err.get("msg", "Invalid value")
                field_errors.append(FieldValidationError(
                    loc=loc_tuple,
                    reason=f"Field '{loc_str}': {msg}",
                    error_code=err.get("type", ValidationErrorCode.SCHEMA_VALIDATION_FAILED.value),
                ))

            primary_err = field_errors[0] if field_errors else None
            reason_msg = primary_err.reason if primary_err else "Schema validation failed"

            rejection = SchemaRejectionPayload(
                status="rejected",
                error_code=ValidationErrorCode.SCHEMA_VALIDATION_FAILED.value,
                reason=reason_msg,
                loc=list(primary_err.loc) if primary_err else ["root"],
                errors=[e.to_dict() for e in field_errors],
            )
            return None, rejection

    @staticmethod
    def validate_diagnosis(
        data: Any,
    ) -> Tuple[Optional[DiagnosisReasoningSchema], Optional[SchemaRejectionPayload]]:
        """Validate diagnosis recommendation and reasoning output.

        Returns:
            (validated_instance, None) on success
            (None, rejection_payload) on validation error
        """
        if not isinstance(data, dict):
            rejection = SchemaRejectionPayload(
                status="rejected",
                error_code=ValidationErrorCode.UNEXPECTED_DATA_SHAPE.value,
                reason=f"Diagnosis data must be a dictionary, received {type(data).__name__}",
                loc=["root"],
                errors=[{"loc": ["root"], "reason": f"Expected dict, got {type(data).__name__}", "error_code": "type_error"}],
            )
            return None, rejection

        try:
            validated = DiagnosisReasoningSchema.model_validate(data)
            return validated, None
        except ValidationError as ve:
            field_errors = []
            for err in ve.errors():
                loc_tuple = tuple(str(x) for x in err.get("loc", []))
                loc_str = ".".join(loc_tuple)
                msg = err.get("msg", "Invalid value")
                field_errors.append(FieldValidationError(
                    loc=loc_tuple,
                    reason=f"Field '{loc_str}': {msg}",
                    error_code=err.get("type", ValidationErrorCode.SCHEMA_VALIDATION_FAILED.value),
                ))

            primary_err = field_errors[0] if field_errors else None
            reason_msg = primary_err.reason if primary_err else "Diagnosis schema validation failed"

            rejection = SchemaRejectionPayload(
                status="rejected",
                error_code=ValidationErrorCode.INVALID_DIAGNOSIS_OUTPUT.value,
                reason=reason_msg,
                loc=list(primary_err.loc) if primary_err else ["root"],
                errors=[e.to_dict() for e in field_errors],
            )
            return None, rejection
