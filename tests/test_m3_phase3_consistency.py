"""
Unit Tests for Phase 3 (M3.3): Clinical Consistency Rule Engine.

Covers:
- M3.3 #1: Laterality mismatch flagged (English & Arabic terms, Left vs Right)
- M3.3 #2: Diagnosis without evidence flagged (Hallucination guard)
- Clean passes when laterality matches or evidence is well grounded
- Sub-millisecond performance (< 5ms per record)
"""

import sys
import time
from pathlib import Path

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from validation.consistency import (
    CautionSeverity,
    ConsistencyRuleType,
    ConsistencyValidator,
)
from validation.laterality import (
    Laterality,
    check_laterality_conflict,
    detect_laterality,
)


def test_laterality_detection_english():
    """Verify laterality extraction from English text."""
    left_det = detect_laterality("Patient complains of severe pain in the left knee.")
    assert left_det.laterality == Laterality.LEFT
    assert "left" in left_det.matched_tokens

    right_det = detect_laterality("Examination shows tenderness in right lower quadrant.")
    assert right_det.laterality == Laterality.RIGHT
    assert "right" in right_det.matched_tokens

    bilateral_det = detect_laterality("Bilateral crackles heard over lower lung bases.")
    assert bilateral_det.laterality == Laterality.BILATERAL
    assert "bilateral" in bilateral_det.matched_tokens


def test_laterality_detection_arabic():
    """Verify laterality extraction from Arabic text."""
    ar_left = detect_laterality("ألم شديد في الركبة اليسرى منذ يومين")
    assert ar_left.laterality == Laterality.LEFT

    ar_right = detect_laterality("تورم في الجانب الأيمن من البطن")
    assert ar_right.laterality == Laterality.RIGHT

    ar_bilateral = detect_laterality("احتقان في كلا الجانبين")
    assert ar_bilateral.laterality == Laterality.BILATERAL


def test_laterality_mismatch_flagged():
    """M3.3 #1: Laterality mismatch between clinical findings and diagnosis is flagged."""
    findings = "Patient presents with persistent severe pain in the left knee upon walking."
    diagnosis_wrong_side = "M17.11 - Unilateral primary osteoarthritis, right knee"

    conflict = check_laterality_conflict(findings, diagnosis_wrong_side)
    assert conflict is not None
    assert "Laterality mismatch detected" in conflict
    assert "left" in conflict and "right" in conflict

    # Test via ConsistencyValidator
    cautions = ConsistencyValidator.check_laterality(findings, [diagnosis_wrong_side])
    assert len(cautions) == 1
    banner = cautions[0]
    assert banner.rule == ConsistencyRuleType.LATERALITY_MISMATCH.value
    assert banner.severity == CautionSeverity.HIGH.value
    assert banner.code_affected == "M17.11"
    assert "left" in banner.message and "right" in banner.message


def test_laterality_match_passes_cleanly():
    """Verify that when laterality matches, no caution is raised."""
    findings = "Pain localized to right knee with reduced range of motion."
    diagnosis_matching_side = "M17.11 - Unilateral primary osteoarthritis, right knee"

    conflict = check_laterality_conflict(findings, diagnosis_matching_side)
    assert conflict is None

    cautions = ConsistencyValidator.check_laterality(findings, [diagnosis_matching_side])
    assert len(cautions) == 0


def test_diagnosis_without_evidence_flagged():
    """M3.3 #2: Diagnosis without evidence in clinical text is flagged."""
    # Clinical text describes only respiratory symptoms
    clinical_text = "Patient presents with dry cough, low-grade fever, and mild sore throat."
    
    # Model hallucinates completely ungrounded appendicitis diagnosis
    hallucinated_dx = "K35.80 - Acute appendicitis, other and unspecified"

    cautions = ConsistencyValidator.check_evidence_grounding(clinical_text, [hallucinated_dx])
    assert len(cautions) == 1
    banner = cautions[0]
    assert banner.rule == ConsistencyRuleType.DIAGNOSIS_UNGROUNDED.value
    assert banner.severity == CautionSeverity.MEDIUM.value
    assert banner.code_affected == "K35.80"
    assert "no documented supporting evidence" in banner.message


def test_grounded_diagnosis_passes_cleanly():
    """Verify that grounded diagnoses pass without ungrounded cautions."""
    clinical_text = (
        "Patient presents with fever and cough for 4 days. Auscultation reveals bilateral crackles. "
        "Chest X-ray shows right lower lobe consolidation consistent with pneumonia."
    )
    grounded_dx = "J18.9 - Pneumonia, unspecified organism"

    cautions = ConsistencyValidator.check_evidence_grounding(clinical_text, [grounded_dx])
    assert len(cautions) == 0


def test_combined_consistency_validation():
    """Test full ConsistencyValidator.validate_consistency method."""
    extraction = {
        "chief_complaint": "Severe swelling and pain in left ankle after twisting during sports.",
        "examination": "Left ankle tender over lateral malleolus.",
    }
    diagnoses = [
        "S93.40 - Sprain of unspecified ligament of right ankle",  # Laterality mismatch!
        "E11.9 - Type 2 diabetes mellitus without complications",  # Ungrounded hallucination!
    ]

    cautions = ConsistencyValidator.validate_consistency(extraction, diagnoses)
    assert len(cautions) == 2

    rules = [c.rule for c in cautions]
    assert ConsistencyRuleType.LATERALITY_MISMATCH.value in rules
    assert ConsistencyRuleType.DIAGNOSIS_UNGROUNDED.value in rules


def test_consistency_validation_performance():
    """Verify consistency checking executes in sub-millisecond time (< 1ms)."""
    extraction = {
        "chief_complaint": "Pain in the left knee with swelling.",
        "examination": "Tenderness on medial joint line of left knee.",
    }
    diagnoses = ["M17.12 - Unilateral primary osteoarthritis, left knee"]

    t0 = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        ConsistencyValidator.validate_consistency(extraction, diagnoses)
    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / iterations

    assert avg_ms < 1.0, f"Average consistency validation time {avg_ms:.4f}ms exceeded 1ms budget"


if __name__ == "__main__":
    test_laterality_detection_english()
    test_laterality_detection_arabic()
    test_laterality_mismatch_flagged()
    test_laterality_match_passes_cleanly()
    test_diagnosis_without_evidence_flagged()
    test_grounded_diagnosis_passes_cleanly()
    test_combined_consistency_validation()
    test_consistency_validation_performance()
    print("All Phase 3 (M3.3) consistency rule tests passed successfully!")
