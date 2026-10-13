"""
evaluation package
==================
Gold Case evaluation engine, metrics calculators, and Excel dataset importers.
"""
from .metrics import (
    FieldScore,
    compare_fields,
    compare_diagnosis,
    calc_accuracy,
    calc_completeness,
    calc_f1,
    fuzzy_similarity,
    compute_composite_score,
)
from .importers import load_gold_cases_from_xlsx
from .engine import EvalResult, EvaluationEngine

__all__ = [
    "FieldScore",
    "compare_fields",
    "compare_diagnosis",
    "calc_accuracy",
    "calc_completeness",
    "calc_f1",
    "fuzzy_similarity",
    "compute_composite_score",
    "load_gold_cases_from_xlsx",
    "EvalResult",
    "EvaluationEngine",
]
