"""Comprehensive Unit Tests for ICD-10 Registry Service.

Covers all Phase 5.1 requirements:
- test_registry_loads_all_codes()        # 16,953 entries
- test_exact_code_lookup()               # "A00.0" -> match
- test_code_case_insensitive()           # "a00.0" -> match
- test_code_with_dot_and_without()       # "A000" and "A00.0" both work
- test_search_by_code_prefix()           # "J18" -> J18.0, J18.1, J18.9, ...
- test_search_by_description()           # "pneumonia" -> multiple results
- test_fuzzy_search_typo_tolerance()     # "pnuemonia" -> still finds pneumonia
- test_validate_codes_exact_match()      # LLM returns valid code -> passes
- test_validate_codes_parent_fallback()  # LLM returns "J18.99" (not in list) -> maps to "J18.9"
- test_validate_codes_description_match()# LLM returns unknown code -> fuzzy match by description
- test_validate_codes_no_match()         # Truly invalid -> marked unmatched
- test_search_performance()              # < 15ms for typical queries
"""

import sys
import time
from pathlib import Path

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.icd10_registry import get_registry, normalize_code


def test_registry_loads_all_codes():
    registry = get_registry()
    assert len(registry) == 16953
    codes = registry.all_codes()
    assert len(codes) == 16953


def test_exact_code_lookup():
    registry = get_registry()
    entry = registry.lookup_code("A00.0")
    assert entry is not None
    assert entry.code == "A00.0"
    assert "Cholera" in entry.description


def test_code_case_insensitive():
    registry = get_registry()
    entry = registry.lookup_code("a00.0")
    assert entry is not None
    assert entry.code == "A00.0"


def test_code_with_dot_and_without():
    registry = get_registry()
    with_dot = registry.lookup_code("A00.0")
    without_dot = registry.lookup_code("A000")
    assert with_dot is not None
    assert without_dot is not None
    assert with_dot.code == without_dot.code == "A00.0"


def test_search_by_code_prefix():
    registry = get_registry()
    results = registry.search("J18", limit=10)
    assert len(results) >= 5
    codes = [r[0] for r in results]
    assert all(c.startswith("J18") for c in codes)
    assert "J18.0" in codes
    assert "J18.9" in codes


def test_search_by_description():
    registry = get_registry()
    results = registry.search("pneumonia", limit=10)
    assert len(results) == 10
    assert any("pneumonia" in desc.lower() for _, desc in results)


def test_fuzzy_search_typo_tolerance():
    registry = get_registry()
    results = registry.search("pnuemonia", limit=5)
    assert len(results) > 0
    # Checks that typo query still retrieves pneumonia-related ICD codes
    assert any("pneumonia" in desc.lower() for _, desc in results)


def test_validate_codes_exact_match():
    registry = get_registry()
    val = registry.validate_code("A00.0 - Cholera due to Vibrio cholerae 01, biovar cholerae")
    assert val.is_valid is True
    assert val.code == "A00.0"
    assert val.matched_entry is not None


def test_validate_codes_parent_fallback():
    registry = get_registry()
    # J18.99 is not in Australian list, should resolve to J18.9
    val = registry.validate_code("J18.99 - Pneumonia condition")
    assert val.is_valid is True
    assert val.code == "J18.9"
    assert val.corrected_label.startswith("J18.9")


def test_validate_codes_description_match():
    registry = get_registry()
    # Pure description with high similarity to official label
    val = registry.validate_code("Acute appendicitis with generalised peritonitis")
    assert val.is_valid is True
    assert val.code == "K35.2"


def test_validate_codes_no_match():
    registry = get_registry()
    val = registry.validate_code("FAKE999 - Entirely fabricated disease condition")
    assert val.is_valid is False
    assert val.matched_entry is None
    assert val.original == "FAKE999 - Entirely fabricated disease condition"


def test_search_performance():
    registry = get_registry()
    queries = ["J18", "pneumonia", "diabetes", "appendicitis", "fracture femur"]
    # Warm up
    for q in queries:
        registry.search(q, limit=10)

    for q in queries:
        t0 = time.perf_counter()
        hits = registry.search(q, limit=10)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        assert latency_ms < 15.0, f"Search for {q} took {latency_ms:.2f} ms (expected < 15ms)"
        assert len(hits) > 0


if __name__ == "__main__":
    test_registry_loads_all_codes()
    test_exact_code_lookup()
    test_code_case_insensitive()
    test_code_with_dot_and_without()
    test_search_by_code_prefix()
    test_search_by_description()
    test_fuzzy_search_typo_tolerance()
    test_validate_codes_exact_match()
    test_validate_codes_parent_fallback()
    test_validate_codes_description_match()
    test_validate_codes_no_match()
    test_search_performance()
    print("All Phase 5.1 ICD-10 registry unit tests passed successfully!")
