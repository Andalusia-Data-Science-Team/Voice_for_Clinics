"""
Unit Tests for Phase 4 (M3.4): Unified History Cross-Check & Deterministic Missing Fields.

Covers:
- M3.4 #1: History cross-check reused and working (overlap detection, claims_used_no_overlap flag)
- M3.4 #2: Missing-field checklist generated deterministically without secondary LLM guessing
- High-performance execution (< 1ms per record)
"""

import sys
import time
from pathlib import Path

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from validation.history_guard import (
    HistoryAlignmentReport,
    HistoryCrossCheckGuard,
    HistoryDiscrepancyFlag,
)
from validation.missing_fields import (
    MissingFieldChecklist,
    MissingFieldsReport,
)


def test_history_cross_check_overlap_success():
    """M3.4 #1: History cross-check recognizes shared diagnostic families."""
    mock_visits = [
        {
            "visit_date": "2025-05-10",
            "icd10_diagnoses": ["E11.9 - Type 2 diabetes mellitus without complications"],
        }
    ]
    extracted_features = {
        "icd10_codes": ["E11.01 - Type 2 diabetes with hyperosmolarity", "J18.9 - Pneumonia"],
        "history_considered": "Yes",
    }

    report = HistoryCrossCheckGuard.evaluate(mock_visits, extracted_features)
    assert report.history_provided is True
    assert report.claimed_considered is True
    assert report.overlap_found is True
    assert "E11" in report.overlap_families
    assert report.flag is None
    assert report.caution_banner is None


def test_history_cross_check_claims_used_no_overlap():
    """M3.4 #1: Flags when model claims history was used but no code families match."""
    mock_visits = [
        {
            "visit_date": "2025-01-15",
            "icd10_diagnoses": ["M17.11 - Unilateral osteoarthritis, right knee"],
        }
    ]
    # Today's diagnosis is purely respiratory with no musculoskeletal overlap
    extracted_features = {
        "icd10_codes": ["J18.9 - Pneumonia, unspecified"],
        "history_considered": "Yes",
    }

    report = HistoryCrossCheckGuard.evaluate(mock_visits, extracted_features)
    assert report.history_provided is True
    assert report.claimed_considered is True
    assert report.overlap_found is False
    assert report.flag == HistoryDiscrepancyFlag.CLAIMS_USED_NO_OVERLAP.value
    assert report.caution_banner is not None
    assert "claimed prior patient history was considered" in report.caution_banner.message


def test_history_cross_check_overlap_but_not_flagged():
    """M3.4 #1: Flags when prior chronic condition matches but history_considered is No."""
    mock_visits = [
        {
            "visit_date": "2025-01-15",
            "icd10_diagnoses": ["I10 - Essential (primary) hypertension"],
        }
    ]
    extracted_features = {
        "icd10_codes": ["I10 - Essential (primary) hypertension"],
        "history_considered": "No",
    }

    report = HistoryCrossCheckGuard.evaluate(mock_visits, extracted_features)
    assert report.history_provided is True
    assert report.claimed_considered is False
    assert report.overlap_found is True
    assert report.flag == HistoryDiscrepancyFlag.OVERLAP_BUT_NOT_FLAGGED.value
    assert report.caution_banner is not None
    assert "I10" in report.caution_banner.message


def test_missing_fields_deterministic_opd():
    """M3.4 #2: Missing-field checklist identifies missing OPD fields deterministically."""
    # Partially filled OPD note
    extracted = {
        "chief_complaint": "Persistent cough",
        "onset": "3 days ago",
        "duration": "3 days",
        "allergy_status": "NKDA",
        "examination_finding": "Normal breath sounds",
        "plan_of_care": "Cough syrup",
    }
    report = MissingFieldChecklist.evaluate(extracted, department="opd")

    assert report.department == "opd"
    assert report.total_fields == 17
    assert report.present_count == 6
    assert report.missing_count == 11
    assert report.completeness_percentage < 100.0

    missing_keys = [f.key for f in report.missing_fields]
    assert "medication_history" in missing_keys
    assert "comorbidities" in missing_keys
    assert "past_medical_history" in missing_keys
    # Present fields must NOT be in missing list
    assert "chief_complaint" not in missing_keys
    assert "plan_of_care" not in missing_keys


def test_missing_fields_deterministic_er():
    """M3.4 #2: Missing-field checklist identifies missing ER fields deterministically."""
    extracted = {
        "chief_complaint": "Acute abdominal pain",
        "airway": "Patent",
        "breathing": "Normal",
        "circulation": "Stable",
        "disability": "Alert",
        "exposure": "Tender RLQ",
    }
    report = MissingFieldChecklist.evaluate(extracted, department="er")

    assert report.department == "er"
    assert report.total_fields == 22
    assert report.present_count == 6
    assert report.missing_count == 16

    missing_keys = [f.key for f in report.missing_fields]
    assert "plan_of_care" in missing_keys
    assert "investigations" in missing_keys
    assert "chief_complaint" not in missing_keys


def test_missing_fields_empty_check_variants():
    """Verify empty/null/N/A variations are correctly classified as missing."""
    extracted = {
        "chief_complaint": "",
        "onset": None,
        "course": "None",
        "duration": "N/A",
        "past_medical_history": "Not stated",
        "medication_history": [],
    }
    report = MissingFieldChecklist.evaluate(extracted, department="opd")
    assert report.present_count == 0
    assert report.missing_count == report.total_fields


def test_phase4_performance():
    """Verify both history guard and missing fields run in sub-millisecond time (< 1ms)."""
    mock_visits = [{"visit_date": "2025-01-01", "icd10_diagnoses": ["E11.9"]}]
    extracted = {"icd10_codes": ["E11.9"], "history_considered": "Yes", "chief_complaint": "Diabetes review"}

    t0 = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        HistoryCrossCheckGuard.evaluate(mock_visits, extracted)
        MissingFieldChecklist.evaluate(extracted, department="opd")
    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / iterations

    assert avg_ms < 1.0, f"Average execution time {avg_ms:.4f}ms exceeded 1ms budget"


if __name__ == "__main__":
    test_history_cross_check_overlap_success()
    test_history_cross_check_claims_used_no_overlap()
    test_history_cross_check_overlap_but_not_flagged()
    test_missing_fields_deterministic_opd()
    test_missing_fields_deterministic_er()
    test_missing_fields_empty_check_variants()
    test_phase4_performance()
    print("All Phase 4 (M3.4) history guard & missing fields tests passed successfully!")
