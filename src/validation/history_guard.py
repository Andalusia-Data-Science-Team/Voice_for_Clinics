"""
MedVoice Validation Subsystem - Patient History Cross-Check Guard
==================================================================
Part of Phase 4 (Milestone 3.4): History cross-check + missing-field checklist.

Features:
  - Strongly typed data structures for history alignment
  - Integrates with model.patient_history.check_history_usage
  - Surfaces actionable discrepancies:
      • claims_used_no_overlap: Model claimed history was used, but zero historical diagnostic families match.
      • overlap_but_not_flagged: Overlap exists with historical chronic condition, but model didn't flag history consideration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from model.patient_history import check_history_usage
from validation.consistency import CautionBanner, CautionSeverity

logger = logging.getLogger(__name__)


class HistoryDiscrepancyFlag(str, Enum):
    """Flags indicating potential misalignment between claimed and actual history usage."""
    CLAIMS_USED_NO_OVERLAP = "claims_used_no_overlap"
    OVERLAP_BUT_NOT_FLAGGED = "overlap_but_not_flagged"


@dataclass(frozen=True)
class HistoryAlignmentReport:
    """Standardized report on patient history correlation."""
    history_provided: bool
    claimed_considered: bool
    overlap_found: bool
    overlap_families: list[str]
    flag: Optional[str] = None
    caution_banner: Optional[CautionBanner] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_provided": self.history_provided,
            "claimed_considered": self.claimed_considered,
            "overlap_found": self.overlap_found,
            "overlap_families": self.overlap_families,
            "flag": self.flag,
            "caution": self.caution_banner.to_dict() if self.caution_banner else None,
        }


class HistoryCrossCheckGuard:
    """High-performance validator for patient history correlation."""

    @classmethod
    def evaluate(
        cls,
        past_visits: Optional[Sequence[dict[str, Any]]],
        extracted_features: Optional[dict[str, Any]],
    ) -> HistoryAlignmentReport:
        """Cross-check past patient visits against today's diagnosis recommendations.

        Reuses check_history_usage while providing clean typing and caution banners.
        """
        visits_list = list(past_visits or [])
        features_dict = dict(extracted_features or {})

        # Execute core historical comparison
        raw_result = check_history_usage(visits_list, features_dict)

        flag = raw_result.get("flag")
        caution: Optional[CautionBanner] = None

        if flag == HistoryDiscrepancyFlag.CLAIMS_USED_NO_OVERLAP.value:
            caution = CautionBanner(
                rule="HISTORY_CROSS_CHECK",
                severity=CautionSeverity.LOW.value,
                message=(
                    "History notice: Clinical recommendation claimed prior patient history was considered, "
                    "but none of the past diagnosis families match today's ICD-10 codes."
                ),
                field_affected="history_considered",
                code_affected=None,
            )
        elif flag == HistoryDiscrepancyFlag.OVERLAP_BUT_NOT_FLAGGED.value:
            families_str = ", ".join(raw_result.get("overlap_families", []))
            caution = CautionBanner(
                rule="HISTORY_CROSS_CHECK",
                severity=CautionSeverity.INFO.value,
                message=(
                    f"History notice: Overlap found with historical chronic condition family ({families_str}), "
                    "but 'history_considered' was not marked as 'Yes'."
                ),
                field_affected="history_considered",
                code_affected=None,
            )

        return HistoryAlignmentReport(
            history_provided=bool(raw_result.get("history_provided", False)),
            claimed_considered=bool(raw_result.get("claimed_considered", False)),
            overlap_found=bool(raw_result.get("overlap_found", False)),
            overlap_families=list(raw_result.get("overlap_families", [])),
            flag=flag,
            caution_banner=caution,
        )
