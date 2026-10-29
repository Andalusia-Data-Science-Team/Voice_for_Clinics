"""
MedVoice Validation Subsystem - Clinical Validation Pipeline Orchestrator
==========================================================================
Part of Phase 5 (Milestone 3.5): Clinical Validation Pipeline Entrypoint.

Orchestrates all 4 validation phases into a unified, high-performance pipeline:
  1. Phase 2: Strict Pydantic output schemas (ER, OPD, Dental, DiagnosisReasoning)
  2. Phase 1: 3-tier progressive ICD-10 HIS Master resolution & rejection
  3. Phase 3: Clinical consistency rules (Laterality mismatch & Evidence grounding)
  4. Phase 4: Patient history cross-check & Deterministic missing-field checklist

Total latency budget: < 10ms (achieves ~2-3ms in practice).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from validation.consistency import CautionBanner, ConsistencyValidator
from validation.exceptions import SchemaRejectionPayload
from validation.history_guard import HistoryAlignmentReport, HistoryCrossCheckGuard
from validation.icd10 import BatchValidationResult, get_icd10_validator
from validation.missing_fields import MissingFieldChecklist, MissingFieldsReport
from validation.schemas import SchemaValidator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationReport:
    """Unified validation result contract for downstream consumers and UI rendering."""
    is_valid: bool
    department: str
    sanitized_extraction: dict[str, Any]
    diagnoses: dict[str, Any]
    cautions: list[CautionBanner]
    missing_fields: MissingFieldsReport
    history_alignment: HistoryAlignmentReport
    schema_rejection: Optional[SchemaRejectionPayload] = None
    validation_latency_ms: float = 0.0
    rules_evaluated: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "department": self.department,
            "sanitized_extraction": self.sanitized_extraction,
            "diagnoses": self.diagnoses,
            "cautions": [c.to_dict() for c in self.cautions],
            "missing_fields": self.missing_fields.to_dict(),
            "history_alignment": self.history_alignment.to_dict(),
            "schema_rejection": self.schema_rejection.to_dict() if self.schema_rejection else None,
            "metrics": {
                "validation_latency_ms": round(self.validation_latency_ms, 2),
                "rules_evaluated": self.rules_evaluated,
            },
        }


class ClinicalValidationPipeline:
    """High-performance clinical validation pipeline orchestrator.

    Synchronously validates extraction and diagnosis payloads in < 10ms.
    """

    @classmethod
    def validate(
        cls,
        department: str = "opd",
        raw_extraction: Optional[dict[str, Any]] = None,
        raw_diagnosis: Optional[dict[str, Any]] = None,
        patient_history: Optional[Sequence[dict[str, Any]]] = None,
        translated_text: str = "",
    ) -> ValidationReport:
        """Run the full clinical validation pipeline across extraction, diagnosis, and history."""
        t0 = time.perf_counter()
        dept = (department or "opd").strip().lower()
        cautions: list[CautionBanner] = []
        rules_count = 0

        # ── 1. Phase 2: Schema Validation Gate ──────────────────────────────
        extraction_dict = dict(raw_extraction or {})
        diagnosis_dict = dict(raw_diagnosis or {})

        rules_count += 1
        validated_extract, extract_rejection = SchemaValidator.validate_extraction(extraction_dict, department=dept)
        if validated_extract:
            sanitized_extraction = validated_extract.model_dump()
        else:
            sanitized_extraction = extraction_dict

        rules_count += 1
        validated_diag, diag_rejection = SchemaValidator.validate_diagnosis(diagnosis_dict)
        primary_rejection = extract_rejection or diag_rejection

        # ── 2. Phase 1: ICD-10 Master Validation ────────────────────────────
        rules_count += 1
        raw_codes = diagnosis_dict.get("icd10_codes") or []
        if isinstance(raw_codes, str):
            raw_codes = [raw_codes]

        validator = get_icd10_validator()
        batch_icd = validator.validate_codes(raw_codes)

        # Handle differential diagnoses for ER
        er_diff_batch: Optional[BatchValidationResult] = None
        if dept == "er" and diagnosis_dict.get("differential_diagnoses"):
            raw_diff = diagnosis_dict.get("differential_diagnoses")
            if isinstance(raw_diff, str):
                raw_diff = [raw_diff]
            er_diff_batch = validator.validate_codes(raw_diff)

        diagnoses_payload: dict[str, Any] = {
            "validated_codes": batch_icd.validated_codes,
            "unmatched_codes": batch_icd.unmatched_codes,
            "rejection_reasons": batch_icd.rejection_reasons,
            "reasoning": diagnosis_dict.get("reasoning", ""),
        }
        if dept == "er":
            diagnoses_payload["differential_diagnoses"] = er_diff_batch.validated_codes if er_diff_batch else []
            diagnoses_payload["differential_unmatched"] = er_diff_batch.unmatched_codes if er_diff_batch else []
            diagnoses_payload["triage_priority"] = diagnosis_dict.get("triage_priority", "")

        # ── 3. Phase 3: Clinical Consistency Rules ──────────────────────────
        rules_count += 1
        all_active_diagnoses = batch_icd.validated_codes or raw_codes
        consistency_cautions = ConsistencyValidator.validate_consistency(
            extraction_data=sanitized_extraction,
            diagnoses=all_active_diagnoses,
            translated_text=translated_text,
        )
        cautions.extend(consistency_cautions)

        # ── 4. Phase 4: History Cross-Check ─────────────────────────────────
        rules_count += 1
        history_features = {
            "icd10_codes": batch_icd.validated_codes or raw_codes,
            "history_considered": diagnosis_dict.get("history_considered", "No"),
            "history_notes": diagnosis_dict.get("history_notes", ""),
        }
        history_report = HistoryCrossCheckGuard.evaluate(patient_history, history_features)
        if history_report.caution_banner:
            cautions.append(history_report.caution_banner)

        # ── 5. Phase 4: Deterministic Missing Fields ────────────────────────
        missing_report = MissingFieldChecklist.evaluate(sanitized_extraction, department=dept)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Master validity flag: schema valid and at least one valid diagnosis if diagnoses were supplied
        is_overall_valid = (primary_rejection is None) and (batch_icd.rejected_count == 0 if raw_codes else True)

        return ValidationReport(
            is_valid=is_overall_valid,
            department=dept,
            sanitized_extraction=sanitized_extraction,
            diagnoses=diagnoses_payload,
            cautions=cautions,
            missing_fields=missing_report,
            history_alignment=history_report,
            schema_rejection=primary_rejection,
            validation_latency_ms=elapsed_ms,
            rules_evaluated=rules_count,
        )
