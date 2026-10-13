"""
tests/test_evaluation.py
=========================
Unit tests for Phase 4 Evaluation Engine and Metrics:
- Field normalization and fuzzy string similarity
- ICD-10 diagnosis matching logic
- Field comparison, accuracy, completeness, and F1 calculations
- Composite scoring formula
- EvaluationEngine evaluation of a session against a gold case
- EvaluationEngine batch runs
"""
from __future__ import annotations
import pytest
import pytest_asyncio
from mongomock_motor import AsyncMongoMockClient

from db.repositories import GoldDatasetRepository, EvalRepository, SessionRepository
from schemas.gold_schemas import GoldCase
from schemas.output_schemas import PipelineSessionOutput
from evaluation.metrics import (
    normalize_val,
    fuzzy_similarity,
    extract_icd10_codes,
    compare_diagnosis,
    compare_fields,
    calc_accuracy,
    calc_completeness,
    calc_f1,
    compute_composite_score,
)
from evaluation.engine import EvaluationEngine, EvalResult


# ==============================================================================
# Metric & comparison unit tests
# ==============================================================================

def test_normalize_val():
    assert normalize_val("  Hello   World \n ") == "hello world"
    assert normalize_val(None) == ""
    assert normalize_val(123) == "123"


def test_fuzzy_similarity():
    assert fuzzy_similarity("Chest pain", "chest pain") == 1.0
    assert fuzzy_similarity("", "") == 1.0
    assert fuzzy_similarity("Chest pain", "") == 0.0
    sim = fuzzy_similarity("Severe chest pain", "Severe chest pains")
    assert sim > 0.90


def test_extract_icd10_codes():
    codes_dict = extract_icd10_codes({"icd10_codes": ["I20.0", "R07.9"]})
    assert codes_dict == {"I200", "R079"}

    codes_str = extract_icd10_codes({"icd10_codes": "I20.0"})
    assert codes_str == {"I200"}

    codes_list = extract_icd10_codes(["I20.0", "i21.9"])
    assert codes_list == {"I200", "I219"}

    assert extract_icd10_codes(None) == set()


def test_compare_diagnosis():
    # Exact ICD-10 overlap
    expected = {"icd10_codes": ["I20.0", "I25.1"]}
    actual = {"icd10_codes": ["I20.0"]}
    assert compare_diagnosis(expected, actual) is True

    # No ICD-10 overlap
    actual_disjoint = {"icd10_codes": ["J45.9"]}
    assert compare_diagnosis(expected, actual_disjoint) is False

    # Empty expected diagnosis defaults to True
    assert compare_diagnosis(None, actual) is True

    # Text diagnosis match when codes omitted
    exp_text = {"final_diagnosis": "Bronchitis"}
    act_text = {"final_diagnosis": "bronchitis "}
    assert compare_diagnosis(exp_text, act_text) is True


def test_compare_fields_and_accuracy():
    expected = {
        "chief_complaint": "Chest pain",
        "onset": "2 hours ago",
        "severity": "Severe",
        "allergies": "NKDA",
    }
    actual = {
        "chief_complaint": "Severe chest pain",  # fuzzy match >= 0.85
        "onset": "2 hours ago",                 # exact match
        "severity": "Mild",                     # mismatch
        "allergies": None,                      # missing
    }

    scores = compare_fields(expected, actual, similarity_threshold=0.70)
    assert len(scores) == 4

    acc = calc_accuracy(scores)
    # chief_complaint: match (sim=0.74 >= 0.70), onset: match (1.0), severity: mismatch, allergies: mismatch -> 2 / 4 = 0.50
    assert acc == 0.50

    comp = calc_completeness(scores)
    # present: chief_complaint, onset, severity (3/4) -> 0.75
    assert comp == 0.75

    f1 = calc_f1(scores)
    assert f1 > 0.50


def test_composite_score():
    score = compute_composite_score(
        accuracy=1.0,
        completeness=1.0,
        diagnosis_match=True,
        avg_similarity=1.0,
    )
    assert score == 1.0

    score_partial = compute_composite_score(
        accuracy=0.8,
        completeness=0.9,
        diagnosis_match=True,
        avg_similarity=0.85,
    )
    assert 0.80 <= score_partial <= 0.95


# ==============================================================================
# EvaluationEngine integration unit tests
# ==============================================================================

@pytest_asyncio.fixture
async def mock_engine_db():
    client = AsyncMongoMockClient()
    db = client["medvoice_eval_test"]
    yield db
    client.close()


@pytest.mark.asyncio
async def test_evaluation_engine_evaluate_session(mock_engine_db):
    gold_repo = GoldDatasetRepository(mock_engine_db)
    eval_repo = EvalRepository(mock_engine_db)
    session_repo = SessionRepository(mock_engine_db)

    # 1. Store gold case
    case = GoldCase(
        case_id="ER-EVAL-01",
        department="er",
        specialty="Cardiology",
        audio_language="ar",
        transcript="مريض يعاني من ألم حاد في الصدر يمتد للكتف الأيسر وضيق تنفس",
        expected_extraction={
            "chief_complaint": "ألم في الصدر وضيق تنفس",
            "vital_signs": "BP 140/90",
        },
        expected_diagnosis={
            "final_diagnosis": "Acute Coronary Syndrome",
            "icd10_codes": ["I20.0"],
        },
    )
    await gold_repo.upsert_case(case)

    # 2. Store session
    session = PipelineSessionOutput(
        session_id="sess-eval-101",
        department="er",
        language="ar",
        mode="doctor",
        transcript="مريض يعاني من ألم حاد في الصدر",
        extracted_features={
            "chief_complaint": "ألم في الصدر وضيق تنفس",
            "vital_signs": "BP 140/90",
        },
        diagnosis={
            "final_diagnosis": "Acute Coronary Syndrome",
            "icd10_codes": ["I20.0"],
            "reasoning": "Classic acute coronary features",
        },
    )
    await session_repo.save_session(session)

    # 3. Run evaluation
    engine = EvaluationEngine(gold_repo, eval_repo, session_repo)
    result: EvalResult = await engine.evaluate_session("sess-eval-101", "ER-EVAL-01")

    assert result.case_id == "ER-EVAL-01"
    assert result.session_id == "sess-eval-101"
    assert result.extraction_accuracy == 1.0
    assert result.extraction_completeness == 1.0
    assert result.diagnosis_match is True
    assert result.total_score >= 0.95

    # 4. Verify persisted eval run in EvalRepository
    persisted_runs = await eval_repo.get_runs_for_case("ER-EVAL-01")
    assert len(persisted_runs) == 1
    assert persisted_runs[0]["eval_id"] == result.eval_id


@pytest.mark.asyncio
async def test_evaluation_engine_run_batch(mock_engine_db):
    gold_repo = GoldDatasetRepository(mock_engine_db)
    eval_repo = EvalRepository(mock_engine_db)
    session_repo = SessionRepository(mock_engine_db)

    # Create 2 gold cases
    for i in range(1, 3):
        case = GoldCase(
            case_id=f"OPD-CASE-{i}",
            department="opd",
            audio_language="en",
            transcript=f"Patient {i} presents with persistent cough and mild fever",
            expected_extraction={"chief_complaint": "Cough", "history": "None"},
        )
        await gold_repo.upsert_case(case)

        # Create session with matching session_id
        session = PipelineSessionOutput(
            session_id=f"OPD-CASE-{i}",
            department="opd",
            language="en",
            mode="doctor",
            extracted_features={"chief_complaint": "Cough", "history": "None"},
        )
        await session_repo.save_session(session)

    engine = EvaluationEngine(gold_repo, eval_repo, session_repo)
    batch_results = await engine.run_batch(department="opd")

    assert len(batch_results) == 2
    assert all(r.extraction_accuracy == 1.0 for r in batch_results)
