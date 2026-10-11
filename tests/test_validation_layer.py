"""
Comprehensive Test Suite for Milestone 3 (M3.5): Clinical Validation Layer.

Contains all 12 dedicated unit tests covering:
  1. test_icd10_exact_and_parent_fallback
  2. test_icd10_reject_hallucinated_code
  3. test_pydantic_schema_valid_opd
  4. test_pydantic_schema_valid_er
  5. test_pydantic_schema_reject_invalid_structure_with_reason
  6. test_laterality_mismatch_flagged
  7. test_laterality_match_passes_cleanly
  8. test_diagnosis_without_evidence_flagged
  9. test_grounded_diagnosis_passes
  10. test_history_cross_check_family_overlap
  11. test_missing_fields_deterministic_generation
  12. test_pipeline_performance_under_10ms
"""

import sys
import time
from pathlib import Path

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from validation import (
    CautionSeverity,
    ClinicalValidationPipeline,
    ConsistencyRuleType,
    ConsistencyValidator,
    ERExtractionSchema,
    HistoryCrossCheckGuard,
    HistoryDiscrepancyFlag,
    MissingFieldChecklist,
    OPDExtractionSchema,
    RejectionReason,
    ResolutionTier,
    SchemaValidator,
    ValidationErrorCode,
    check_laterality_conflict,
    get_icd10_validator,
)


# ---------------------------------------------------------------------------
# Test 1: ICD-10 Exact & Parent Fallback
# ---------------------------------------------------------------------------
def test_icd10_exact_and_parent_fallback():
    """Verify Tier 1 exact match and Tier 2 parent category fallback."""
    validator = get_icd10_validator()

    # Exact match
    res_exact = validator.validate_code("A00.0 - Cholera due to Vibrio cholerae 01")
    assert res_exact.is_valid is True
    assert res_exact.canonical_code == "A00.0"
    assert res_exact.tier == ResolutionTier.EXACT_MATCH

    # Parent fallback: J18.99 (invalid subcode) rolls back to J18.9
    res_fallback = validator.validate_code("J18.99 - Pneumonia condition")
    assert res_fallback.is_valid is True
    assert res_fallback.canonical_code == "J18.9"
    assert res_fallback.tier == ResolutionTier.PARENT_FALLBACK


# ---------------------------------------------------------------------------
# Test 2: ICD-10 Reject Hallucinated Code
# ---------------------------------------------------------------------------
def test_icd10_reject_hallucinated_code():
    """Verify non-existent/hallucinated codes are rejected with structured reason."""
    validator = get_icd10_validator()

    res = validator.validate_code("FAKE999.00 - Hallucinated condition name")
    assert res.is_valid is False
    assert res.canonical_code is None
    assert res.tier == ResolutionTier.UNRESOLVED
    assert res.rejection_reason in (
        RejectionReason.CODE_NOT_FOUND_IN_HIS_MASTER.value,
        RejectionReason.HALLUCINATED_CODE.value,
    )


# ---------------------------------------------------------------------------
# Test 3: Pydantic Schema Valid OPD
# ---------------------------------------------------------------------------
def test_pydantic_schema_valid_opd():
    """Verify valid OPD extraction data parses into OPDExtractionSchema."""
    data = {
        "chief_complaint": "Persistent cough and low-grade fever",
        "onset": "4 days ago",
        "duration": "4 days",
        "comorbidities": "Type 2 diabetes mellitus",
        "allergy_status": "NKDA",
        "plan_of_care": "Chest X-ray and Azithromycin course",
    }
    model, rejection = SchemaValidator.validate_extraction(data, department="opd")
    assert rejection is None
    assert model is not None
    assert isinstance(model, OPDExtractionSchema)
    assert model.chief_complaint == "Persistent cough and low-grade fever"
    assert model.comorbidities == "Type 2 diabetes mellitus"


# ---------------------------------------------------------------------------
# Test 4: Pydantic Schema Valid ER
# ---------------------------------------------------------------------------
def test_pydantic_schema_valid_er():
    """Verify valid ER extraction parses primary survey (ABCDE) into ERExtractionSchema."""
    data = {
        "chief_complaint": "Acute severe right lower quadrant pain",
        "airway": "Patent",
        "breathing": "Symmetrical, 98% on room air",
        "circulation": "BP 130/80, HR 88",
        "disability": "GCS 15",
        "exposure": "McBurney point tenderness",
        "plan_of_care": "Laparoscopic appendectomy",
    }
    model, rejection = SchemaValidator.validate_extraction(data, department="er")
    assert rejection is None
    assert model is not None
    assert isinstance(model, ERExtractionSchema)
    assert model.airway == "Patent"
    assert model.exposure == "McBurney point tenderness"


# ---------------------------------------------------------------------------
# Test 5: Pydantic Schema Reject Invalid Structure with Reason
# ---------------------------------------------------------------------------
def test_pydantic_schema_reject_invalid_structure_with_reason():
    """Verify malformed non-dictionary payloads are rejected with reason and location."""
    model, rejection = SchemaValidator.validate_extraction(["Not a dictionary payload"], department="opd")
    assert model is None
    assert rejection is not None
    assert rejection.status == "rejected"
    assert rejection.error_code == ValidationErrorCode.UNEXPECTED_DATA_SHAPE.value
    assert "must be a dictionary" in rejection.reason
    assert rejection.loc == ["root"]


# ---------------------------------------------------------------------------
# Test 6: Laterality Mismatch Flagged
# ---------------------------------------------------------------------------
def test_laterality_mismatch_flagged():
    """Verify anatomical discrepancy (e.g. left knee vs right knee) is flagged."""
    findings = "Patient reports persistent throbbing pain in the left knee."
    diagnosis_wrong_side = "M17.11 - Unilateral primary osteoarthritis, right knee"

    conflict_msg = check_laterality_conflict(findings, diagnosis_wrong_side)
    assert conflict_msg is not None
    assert "Laterality mismatch detected" in conflict_msg

    cautions = ConsistencyValidator.check_laterality(findings, [diagnosis_wrong_side])
    assert len(cautions) == 1
    assert cautions[0].rule == ConsistencyRuleType.LATERALITY_MISMATCH.value
    assert cautions[0].severity == CautionSeverity.HIGH.value


# ---------------------------------------------------------------------------
# Test 7: Laterality Match Passes Cleanly
# ---------------------------------------------------------------------------
def test_laterality_match_passes_cleanly():
    """Verify that consistent laterality produces zero conflict warnings."""
    findings = "Right shoulder tenderness with reduced abduction."
    diagnosis_right = "M75.1 - Rotator cuff syndrome of right shoulder"

    conflict_msg = check_laterality_conflict(findings, diagnosis_right)
    assert conflict_msg is None

    cautions = ConsistencyValidator.check_laterality(findings, [diagnosis_right])
    assert len(cautions) == 0


# ---------------------------------------------------------------------------
# Test 8: Diagnosis Without Evidence Flagged
# ---------------------------------------------------------------------------
def test_diagnosis_without_evidence_flagged():
    """Verify ungrounded diagnosis (hallucination guard) is flagged."""
    clinical_text = "Patient presents with headache and nasal congestion."
    hallucinated_dx = "K35.80 - Acute appendicitis, other and unspecified"

    cautions = ConsistencyValidator.check_evidence_grounding(clinical_text, [hallucinated_dx])
    assert len(cautions) == 1
    assert cautions[0].rule == ConsistencyRuleType.DIAGNOSIS_UNGROUNDED.value
    assert "no documented supporting evidence" in cautions[0].message


# ---------------------------------------------------------------------------
# Test 9: Grounded Diagnosis Passes
# ---------------------------------------------------------------------------
def test_grounded_diagnosis_passes():
    """Verify grounded diagnosis with supporting symptoms passes cleanly."""
    clinical_text = "Productive cough, high fever, and auscultatory crackles consistent with pneumonia."
    grounded_dx = "J18.9 - Pneumonia, unspecified organism"

    cautions = ConsistencyValidator.check_evidence_grounding(clinical_text, [grounded_dx])
    assert len(cautions) == 0


# ---------------------------------------------------------------------------
# Test 10: History Cross-Check Family Overlap
# ---------------------------------------------------------------------------
def test_history_cross_check_family_overlap():
    """Verify history cross-check matches 3-character diagnostic families."""
    past_visits = [
        {"visit_date": "2025-02-10", "icd10_diagnoses": ["E11.9 - Type 2 diabetes without complications"]}
    ]
    extracted_features = {
        "icd10_codes": ["E11.01 - Type 2 diabetes with hyperosmolarity"],
        "history_considered": "Yes",
    }
    report = HistoryCrossCheckGuard.evaluate(past_visits, extracted_features)
    assert report.history_provided is True
    assert report.overlap_found is True
    assert "E11" in report.overlap_families
    assert report.flag is None


# ---------------------------------------------------------------------------
# Test 11: Missing Fields Deterministic Generation
# ---------------------------------------------------------------------------
def test_missing_fields_deterministic_generation():
    """Verify missing fields are generated deterministically with zero LLM guessing."""
    extracted = {
        "chief_complaint": "Persistent cough",
        "onset": "2 days ago",
        "plan_of_care": "Antitussive medication",
    }
    report = MissingFieldChecklist.evaluate(extracted, department="opd")
    assert report.total_fields == 17
    assert report.present_count == 3
    assert report.missing_count == 14
    missing_keys = [f.key for f in report.missing_fields]
    assert "allergy_status" in missing_keys
    assert "chief_complaint" not in missing_keys


# ---------------------------------------------------------------------------
# Test 12: Pipeline Performance Under 10ms
# ---------------------------------------------------------------------------
def test_pipeline_performance_under_10ms():
    """Verify the entire end-to-end pipeline validates under the 10ms budget."""
    raw_extract = {
        "chief_complaint": "Right knee pain after falling",
        "examination": "Right knee swelling and medial tenderness",
        "plan_of_care": "Knee brace and NSAIDs",
        "allergy_status": "NKDA",
    }
    raw_diag = {
        "icd10_codes": ["M17.11 - Unilateral primary osteoarthritis, right knee"],
        "reasoning": "Right knee pain and examination support degenerative joint disease.",
        "history_considered": "No",
    }

    # Warm-up
    for _ in range(10):
        ClinicalValidationPipeline.validate("opd", raw_extract, raw_diag)

    t0 = time.perf_counter()
    report = ClinicalValidationPipeline.validate("opd", raw_extract, raw_diag)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert report.is_valid is True
    assert report.validation_latency_ms < 10.0, f"Pipeline took {report.validation_latency_ms:.2f}ms"
    assert latency_ms < 10.0, f"Measured latency {latency_ms:.2f}ms exceeded 10ms SLA"


def main():
    print("=" * 65)
    print("  MEDVOICE M3 CLINICAL VALIDATION LAYER - 12 UNIT TESTS")
    print("=" * 65)

    test_icd10_exact_and_parent_fallback()
    print("  [PASS] 1. test_icd10_exact_and_parent_fallback")

    test_icd10_reject_hallucinated_code()
    print("  [PASS] 2. test_icd10_reject_hallucinated_code")

    test_pydantic_schema_valid_opd()
    print("  [PASS] 3. test_pydantic_schema_valid_opd")

    test_pydantic_schema_valid_er()
    print("  [PASS] 4. test_pydantic_schema_valid_er")

    test_pydantic_schema_reject_invalid_structure_with_reason()
    print("  [PASS] 5. test_pydantic_schema_reject_invalid_structure_with_reason")

    test_laterality_mismatch_flagged()
    print("  [PASS] 6. test_laterality_mismatch_flagged")

    test_laterality_match_passes_cleanly()
    print("  [PASS] 7. test_laterality_match_passes_cleanly")

    test_diagnosis_without_evidence_flagged()
    print("  [PASS] 8. test_diagnosis_without_evidence_flagged")

    test_grounded_diagnosis_passes()
    print("  [PASS] 9. test_grounded_diagnosis_passes")

    test_history_cross_check_family_overlap()
    print("  [PASS] 10. test_history_cross_check_family_overlap")

    test_missing_fields_deterministic_generation()
    print("  [PASS] 11. test_missing_fields_deterministic_generation")

    test_pipeline_performance_under_10ms()
    print("  [PASS] 12. test_pipeline_performance_under_10ms")

    print("=" * 65)
    print("ALL 12 VALIDATION LAYER TESTS PASSED SUCCESSFULLY! [12/12]")
    print("=" * 65)


if __name__ == "__main__":
    main()
