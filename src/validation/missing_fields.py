"""
MedVoice Validation Subsystem - Deterministic Missing-Field Checklist
======================================================================
Part of Phase 4 (Milestone 3.4): History cross-check + missing-field checklist.

Features:
  - 100% Deterministic evaluation from extracted clinical features
  - Zero secondary LLM guessing (guarantees zero UI card contradictions)
  - Evaluates department checklists (ER, OPD, Dental) with O(1) set operations
  - Returns structured missing-field items with prompt questions for the doctor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from model.department_fields import get_department_fields

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MissingFieldItem:
    """Represents a specific missing clinical field."""
    key: str
    label: str
    question: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "label": self.label,
            "question": self.question,
        }


@dataclass(frozen=True, slots=True)
class MissingFieldsReport:
    """Standardized report on missing fields in the clinical note."""
    department: str
    total_fields: int
    present_count: int
    missing_count: int
    missing_fields: list[MissingFieldItem]
    completeness_percentage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "department": self.department,
            "total_fields": self.total_fields,
            "present_count": self.present_count,
            "missing_count": self.missing_count,
            "completeness_percentage": round(self.completeness_percentage, 1),
            "missing_fields": [f.to_dict() for f in self.missing_fields],
        }


def _is_empty_value(value: Any) -> bool:
    """Determine whether a field value is clinically considered empty."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    val_str = str(value).strip()
    return val_str == "" or val_str.lower() in ("none", "null", "not stated", "not mentioned", "n/a")


class MissingFieldChecklist:
    """High-performance deterministic missing-field generator."""

    @classmethod
    def evaluate(
        cls,
        extracted_features: Optional[dict[str, Any]],
        department: str = "opd",
    ) -> MissingFieldsReport:
        """Deterministically identify which clinical note fields are missing.

        Operates directly on extracted_features without any LLM invocations.
        """
        dept = (department or "opd").strip().lower()
        fields = get_department_fields(dept)
        src = extracted_features or {}

        missing: list[MissingFieldItem] = []
        present_count = 0

        for f in fields:
            val = src.get(f["key"])
            if _is_empty_value(val):
                missing.append(MissingFieldItem(
                    key=f["key"],
                    label=f["label"],
                    question=f["question"],
                ))
            else:
                present_count += 1

        total = len(fields)
        completeness = (present_count / total * 100.0) if total > 0 else 100.0

        return MissingFieldsReport(
            department=dept,
            total_fields=total,
            present_count=present_count,
            missing_count=len(missing),
            missing_fields=missing,
            completeness_percentage=completeness,
        )
