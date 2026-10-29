"""
MedVoice Validation Subsystem - Clinical Consistency & Evidence Grounding Engine
================================================================================
Part of Phase 3 (Milestone 3.3): Consistency rules (laterality, evidence).

Provides two clinical safety rules:
  1. Laterality Guard (LATERALITY_MISMATCH):
     Flags discrepancies between anatomical side stated in symptoms/exam vs ICD-10 code.
  2. Evidence Grounding Guard (DIAGNOSIS_UNGROUNDED):
     Verifies whether each recommended diagnosis has grounding tokens/symptoms in
     the documented clinical encounter, preventing LLM hallucinations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

from model.icd10_registry import tokenize_text
from validation.laterality import check_laterality_conflict, detect_laterality

logger = logging.getLogger(__name__)


class CautionSeverity(str, Enum):
    """Severity levels for clinical caution banners."""
    HIGH = "HIGH"          # Critical contradiction (e.g. wrong side operated/diagnosed)
    MEDIUM = "MEDIUM"      # Diagnosis missing clear grounding evidence
    LOW = "LOW"            # Mild ambiguity or unilateral vs bilateral nuance
    INFO = "INFO"          # Informational alignment note


class ConsistencyRuleType(str, Enum):
    """Rule type enumeration."""
    LATERALITY_MISMATCH = "LATERALITY_MISMATCH"
    DIAGNOSIS_UNGROUNDED = "DIAGNOSIS_UNGROUNDED"


@dataclass(frozen=True)
class CautionBanner:
    """Represents a clinical caution banner to display in the doctor's UI."""
    rule: str
    severity: str
    message: str
    field_affected: Optional[str] = None
    code_affected: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "field_affected": self.field_affected,
            "code_affected": self.code_affected,
        }


# High-frequency clinical anchor tokens to filter out from evidence matching
_COMMON_STOP_TERMS = frozenset({
    "patient", "history", "examination", "finding", "findings", "clinical",
    "disease", "disorder", "syndrome", "acute", "chronic", "unspecified",
    "specified", "other", "primary", "secondary", "due", "without", "with",
    "status", "type", "site", "condition", "illness", "present", "presented"
})


def _extract_diagnosis_clinical_anchor_tokens(code_label: str) -> set[str]:
    """Extract substantive diagnostic clinical concept words from an ICD-10 label."""
    # Label is formatted: "M17.11 - Unilateral primary osteoarthritis, right knee"
    desc = code_label.split(" - ", 1)[-1] if " - " in code_label else code_label
    raw_tokens = tokenize_text(desc)
    # Filter out common medical administrative words
    anchors = {t for t in raw_tokens if t not in _COMMON_STOP_TERMS and len(t) >= 4}
    return anchors


class ConsistencyValidator:
    """High-performance clinical consistency and evidence grounding engine.

    Executes in < 5ms:
    - Zero external I/O
    - Set intersections and regex token scanning
    """

    @classmethod
    def check_laterality(
        cls,
        clinical_text: str,
        diagnoses: Sequence[str],
    ) -> list[CautionBanner]:
        """Rule 1: Verify laterality consistency across clinical findings and diagnoses."""
        cautions: list[CautionBanner] = []
        if not clinical_text or not diagnoses:
            return cautions

        for diag in diagnoses:
            conflict_msg = check_laterality_conflict(clinical_text, diag)
            if conflict_msg:
                code_id = diag.split(" - ", 1)[0].strip() if " - " in diag else diag
                severity = CautionSeverity.HIGH.value if "Laterality mismatch" in conflict_msg else CautionSeverity.MEDIUM.value
                cautions.append(CautionBanner(
                    rule=ConsistencyRuleType.LATERALITY_MISMATCH.value,
                    severity=severity,
                    message=f"Caution: {conflict_msg}",
                    field_affected="examination",
                    code_affected=code_id,
                ))

        return cautions

    @classmethod
    def check_evidence_grounding(
        cls,
        clinical_text: str,
        diagnoses: Sequence[str],
    ) -> list[CautionBanner]:
        """Rule 2: Verify that each diagnosis has documented supporting evidence in the text.

        If a diagnosis contains substantive clinical tokens (e.g. 'appendicitis', 'fracture', 'pneumonia')
        that have ZERO overlap with the clinical text, flag as DIAGNOSIS_UNGROUNDED.
        """
        cautions: list[CautionBanner] = []
        if not clinical_text or not diagnoses:
            return cautions

        text_tokens = set(tokenize_text(clinical_text))
        text_lower = clinical_text.lower()

        for diag in diagnoses:
            anchors = _extract_diagnosis_clinical_anchor_tokens(diag)
            if not anchors:
                continue

            # Check if any anchor word or its prefix is grounded in the text
            has_grounding = False
            for anchor in anchors:
                if anchor in text_tokens or anchor[:5] in text_lower or anchor in text_lower:
                    has_grounding = True
                    break

            if not has_grounding:
                code_id = diag.split(" - ", 1)[0].strip() if " - " in diag else diag
                cautions.append(CautionBanner(
                    rule=ConsistencyRuleType.DIAGNOSIS_UNGROUNDED.value,
                    severity=CautionSeverity.MEDIUM.value,
                    message=(
                        f"Caution: Diagnosis '{diag}' has no documented supporting evidence "
                        f"(e.g. none of {', '.join(sorted(anchors)[:3])}) found in symptoms, physical exam, or history."
                    ),
                    field_affected="chief_complaint",
                    code_affected=code_id,
                ))

        return cautions

    @classmethod
    def validate_consistency(
        cls,
        extraction_data: dict[str, Any],
        diagnoses: Sequence[str],
        translated_text: str = "",
    ) -> list[CautionBanner]:
        """Execute all consistency rules over extracted findings and diagnoses."""
        # Gather all clinical text from extraction plus translated text
        text_parts: list[str] = [translated_text]
        for key in (
            "chief_complaint", "examination", "examination_finding",
            "airway", "breathing", "circulation", "disability", "exposure",
            "past_medical_history", "comorbidities", "investigations"
        ):
            val = extraction_data.get(key)
            if val:
                text_parts.append(str(val))

        aggregated_text = " \n ".join(p for p in text_parts if p)

        cautions: list[CautionBanner] = []
        cautions.extend(cls.check_laterality(aggregated_text, diagnoses))
        cautions.extend(cls.check_evidence_grounding(aggregated_text, diagnoses))
        return cautions
