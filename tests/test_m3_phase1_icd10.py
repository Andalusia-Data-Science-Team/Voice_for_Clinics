"""
Unit Tests for Phase 1 (M3.1): High-Performance ICD-10 Validation & Master Lookup.

Covers:
- M3.1 #1: ICD-10 code master loaded as lookup (16,953 Australian entries)
- M3.1 #2: Invalid code rejected with structured reason
- 3-tier progressive resolution (Exact, Parent, Description fuzzy)
- Performance validation (< 10ms batch throughput)
"""

import sys
import time
from pathlib import Path

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from validation.icd10 import (
    ICD10Validator,
    RejectionReason,
    ResolutionTier,
    get_icd10_validator,
)


def test_validator_loads_all_master_codes():
    """Verify HIS Master ICD-10 registry loads completely and immutably."""
    validator = get_icd10_validator()
    registry = validator.registry
    assert len(registry) == 16953
    all_codes = registry.all_codes()
    assert len(all_codes) == 16953


def test_exact_code_resolution_tier():
    """Tier 1: Exact code match returns EXACT_MATCH tier."""
    validator = get_icd10_validator()
    res = validator.validate_code("A00.0 - Cholera due to Vibrio cholerae 01, biovar cholerae")
    assert res.is_valid is True
    assert res.canonical_code == "A00.0"
    assert res.tier == ResolutionTier.EXACT_MATCH
    assert res.rejection_reason is None
    assert "Cholera" in res.display_label


def test_parent_fallback_resolution_tier():
    """Tier 2: Subcategory not in list rolls back to canonical parent."""
    validator = get_icd10_validator()
    res = validator.validate_code("J18.99 - Pneumonia condition")
    assert res.is_valid is True
    assert res.canonical_code == "J18.9"
    assert res.tier == ResolutionTier.PARENT_FALLBACK
    assert res.rejection_reason is None


def test_description_match_resolution_tier():
    """Tier 3: Clinical description matches official entry with high similarity."""
    validator = get_icd10_validator()
    res = validator.validate_code("Acute appendicitis with generalised peritonitis")
    assert res.is_valid is True
    assert res.canonical_code == "K35.2"
    assert res.tier == ResolutionTier.DESCRIPTION_MATCH
    assert res.rejection_reason is None


def test_invalid_code_rejection_with_reason():
    """Verify invalid/hallucinated codes are rejected with structured reason."""
    validator = get_icd10_validator()
    
    # 1. Hallucinated / fake code
    fake_res = validator.validate_code("FAKE999.00 - Hallucinated condition name")
    assert fake_res.is_valid is False
    assert fake_res.canonical_code is None
    assert fake_res.tier == ResolutionTier.UNRESOLVED
    assert fake_res.rejection_reason in (
        RejectionReason.CODE_NOT_FOUND_IN_HIS_MASTER.value,
        RejectionReason.HALLUCINATED_CODE.value,
    )
    
    # 2. Non-existent alphanumeric code
    xyz_res = validator.validate_code("XYZ.123 - Non-existent medical category")
    assert xyz_res.is_valid is False
    assert xyz_res.rejection_reason is not None

    # 3. Empty input
    empty_res = validator.validate_code("")
    assert empty_res.is_valid is False
    assert empty_res.rejection_reason == RejectionReason.EMPTY_INPUT.value


def test_batch_validation_isolation_and_reasons():
    """Batch validation isolates valid and unmatched codes cleanly."""
    validator = get_icd10_validator()
    raw_list = [
        "A00.0 - Cholera due to Vibrio cholerae",
        "FAKE999.00 - Hallucinated condition",
        "J18.9 - Pneumonia, unspecified",
        "NONEXISTENT.77 - Invalid",
    ]
    batch = validator.validate_codes(raw_list)
    assert batch.valid_count == 2
    assert batch.rejected_count == 2
    assert any("A00.0" in c for c in batch.validated_codes)
    assert any("J18.9" in c for c in batch.validated_codes)
    assert "FAKE999.00 - Hallucinated condition" in batch.unmatched_codes
    assert "NONEXISTENT.77 - Invalid" in batch.unmatched_codes
    assert "FAKE999.00 - Hallucinated condition" in batch.rejection_reasons


def test_batch_validation_performance():
    """Verify batch validation executes within the high-performance SLA (< 10ms)."""
    validator = get_icd10_validator()
    candidates = [
        "A00.0 - Cholera due to Vibrio cholerae",
        "E11.9 - Type 2 diabetes mellitus without complications",
        "I10 - Essential (primary) hypertension",
        "J18.99 - Pneumonia fallback",
        "M17.11 - Unilateral primary osteoarthritis, right knee",
    ] * 4  # 20 items

    # Warm-up
    validator.validate_codes(candidates)

    t0 = time.perf_counter()
    batch = validator.validate_codes(candidates)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert batch.valid_count == 20
    assert latency_ms < 10.0, f"Validation of 20 items took {latency_ms:.2f}ms (expected < 10ms)"


if __name__ == "__main__":
    test_validator_loads_all_master_codes()
    test_exact_code_resolution_tier()
    test_parent_fallback_resolution_tier()
    test_description_match_resolution_tier()
    test_invalid_code_rejection_with_reason()
    test_batch_validation_isolation_and_reasons()
    test_batch_validation_performance()
    print("All Phase 1 (M3.1) tests passed successfully!")
