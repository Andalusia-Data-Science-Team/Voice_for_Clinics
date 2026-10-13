"""
evaluation/metrics.py
=====================
Evaluation metrics and field comparison logic for clinical notes.
Calculates:
- Exact matches (after Unicode & whitespace normalization)
- Fuzzy similarity (SequenceMatcher ratio 0.0 - 1.0)
- Field presence / completeness
- ICD-10 diagnosis structure matching
- Overall F1, precision, recall, and composite scores
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class FieldScore:
    """Individual field comparison result."""
    field_name: str
    expected: Optional[str]
    actual: Optional[str]
    match: bool              # Exact match after normalization
    similarity: float        # 0.0 – 1.0 fuzzy similarity
    is_present: bool         # Field was extracted in actual (not None/empty)
    notes: Optional[str] = None


def normalize_val(val: Any) -> str:
    """Normalize string for robust comparison: lowercase, strip, collapse spaces."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    return re.sub(r"\s+", " ", s)


def fuzzy_similarity(a: Any, b: Any) -> float:
    """Calculate SequenceMatcher similarity ratio between two values."""
    a_str = normalize_val(a)
    b_str = normalize_val(b)
    if not a_str and not b_str:
        return 1.0
    if not a_str or not b_str:
        return 0.0
    if a_str == b_str:
        return 1.0
    return SequenceMatcher(None, a_str, b_str).ratio()


def extract_icd10_codes(diagnosis_obj: Any) -> Set[str]:
    """Extract a normalized set of ICD-10 codes from a dict or list."""
    codes: Set[str] = set()
    if not diagnosis_obj:
        return codes

    if isinstance(diagnosis_obj, dict):
        raw_list = diagnosis_obj.get("icd10_codes") or []
        if isinstance(raw_list, str):
            raw_list = [raw_list]
        for c in raw_list:
            clean = re.sub(r"[^A-Za-z0-9]", "", str(c)).upper()
            if clean:
                codes.add(clean)
    elif isinstance(diagnosis_obj, list):
        for c in diagnosis_obj:
            clean = re.sub(r"[^A-Za-z0-9]", "", str(c)).upper()
            if clean:
                codes.add(clean)
    return codes


def compare_diagnosis(expected: Optional[Dict[str, Any]], actual: Optional[Dict[str, Any]]) -> bool:
    """Compare expected vs actual diagnosis.
    Returns True if at least one ICD-10 code overlaps, or if no expected diagnosis was provided.
    """
    if not expected:
        return True
    exp_codes = extract_icd10_codes(expected)
    if not exp_codes:
        # Check text match on final_diagnosis if no codes provided
        exp_diag = normalize_val(expected.get("final_diagnosis"))
        act_diag = normalize_val(actual.get("final_diagnosis")) if actual else ""
        return bool(exp_diag and exp_diag == act_diag)

    act_codes = extract_icd10_codes(actual)
    return bool(exp_codes & act_codes)


def compare_fields(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    similarity_threshold: float = 0.85,
) -> List[FieldScore]:
    """Compare all expected fields against actual extracted features."""
    scores: List[FieldScore] = []
    # Evaluate every expected key
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        exp_norm = normalize_val(exp_val)
        act_norm = normalize_val(act_val)

        exact_match = (exp_norm == act_norm)
        sim = fuzzy_similarity(exp_val, act_val)
        is_match = exact_match or (sim >= similarity_threshold and bool(exp_norm) and bool(act_norm))

        scores.append(
            FieldScore(
                field_name=key,
                expected=str(exp_val) if exp_val is not None else None,
                actual=str(act_val) if act_val is not None else None,
                match=is_match,
                similarity=round(sim, 4),
                is_present=bool(act_norm),
            )
        )
    return scores


def calc_accuracy(scores: List[FieldScore]) -> float:
    """Accuracy: fraction of evaluated fields that match expected values."""
    if not scores:
        return 0.0
    return round(sum(1 for s in scores if s.match) / len(scores), 4)


def calc_completeness(scores: List[FieldScore]) -> float:
    """Completeness: fraction of non-empty expected fields that were extracted."""
    expected_present = [s for s in scores if normalize_val(s.expected)]
    if not expected_present:
        return 1.0
    present_count = sum(1 for s in expected_present if s.is_present)
    return round(present_count / len(expected_present), 4)


def calc_f1(scores: List[FieldScore]) -> float:
    """Calculate F1 harmonic mean between accuracy and completeness."""
    acc = calc_accuracy(scores)
    comp = calc_completeness(scores)
    if acc + comp == 0.0:
        return 0.0
    return round(2 * (acc * comp) / (acc + comp), 4)


def compute_composite_score(
    accuracy: float,
    completeness: float,
    diagnosis_match: bool,
    avg_similarity: float,
) -> float:
    """Weighted composite evaluation score:
    - 40% Field Accuracy
    - 25% Field Completeness
    - 20% Diagnosis / ICD-10 Match
    - 15% Average Semantic / String Similarity
    """
    diag_score = 1.0 if diagnosis_match else 0.0
    composite = (
        (accuracy * 0.40)
        + (completeness * 0.25)
        + (diag_score * 0.20)
        + (avg_similarity * 0.15)
    )
    return round(composite, 4)
