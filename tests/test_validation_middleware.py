"""
tests/test_validation_middleware.py
===================================
Unit and integration tests for src/validation/middleware.py (Phase 2).
Tests:
- validate_analyze_input dependency (valid payloads, missing fields, cross-field validation, normalization)
- validate_refine_field_input dependency (valid payloads, empty text handling)
- validate_feedback_input dependency (text-only, rating-only, neither raises 422)
- validate_extraction_output (never-fail coercion for ER, OPD, Dental)
- validate_diagnosis_output (never-fail coercion)
"""

import json

import pytest
from fastapi import HTTPException

from schemas.input_schemas import (
    AnalyzeInput,
    BusinessUnit,
    Department,
    FeedbackInput,
    Language,
    Mode,
    RefineFieldInput,
)
from validation.middleware import (
    validate_analyze_input,
    validate_diagnosis_output,
    validate_extraction_output,
    validate_feedback_input,
    validate_refine_field_input,
)


# ==============================================================================
# validate_analyze_input
# ==============================================================================

@pytest.mark.asyncio
async def test_validate_analyze_input_valid():
    inp = await validate_analyze_input(
        text="Patient presents with acute headache",
        language="en",
        mode="doctor",
        department="opd",
        specialty="Cardiology",
        use_history=False,
    )
    assert isinstance(inp, AnalyzeInput)
    assert inp.text == "Patient presents with acute headache"
    assert inp.department == Department.opd
    assert inp.language == Language.english
    assert inp.mode == Mode.doctor


@pytest.mark.asyncio
async def test_validate_analyze_input_missing_text():
    with pytest.raises(HTTPException) as exc_info:
        await validate_analyze_input(
            text="",
            language="en",
            mode="doctor",
            department="opd",
        )
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_validate_analyze_input_use_history_requires_patient_and_bu():
    # use_history=True without patient_id or business_unit must raise 422
    with pytest.raises(HTTPException) as exc_info:
        await validate_analyze_input(
            text="Valid consultation note",
            use_history=True,
            patient_id=None,
            business_unit=None,
        )
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_validate_analyze_input_use_history_success():
    inp = await validate_analyze_input(
        text="Valid consultation note",
        use_history=True,
        patient_id="PAT12345",
        business_unit="AHJ",
    )
    assert inp.use_history is True
    assert inp.patient_id == "PAT12345"
    assert inp.business_unit == BusinessUnit.AHJ


# ==============================================================================
# validate_refine_field_input
# ==============================================================================

@pytest.mark.asyncio
async def test_validate_refine_field_input_valid():
    inp = await validate_refine_field_input(
        text="Doctor notes no allergies",
        language="en",
        specialty="General",
    )
    assert isinstance(inp, RefineFieldInput)
    assert inp.text == "Doctor notes no allergies"


@pytest.mark.asyncio
async def test_validate_refine_field_input_empty_text():
    with pytest.raises(HTTPException) as exc_info:
        await validate_refine_field_input(
            text="   ",
            language="en",
        )
    assert exc_info.value.status_code == 422


# ==============================================================================
# validate_feedback_input
# ==============================================================================

@pytest.mark.asyncio
async def test_validate_feedback_input_valid_text():
    inp = await validate_feedback_input(
        text="Great transcription accuracy",
        rating=None,
    )
    assert isinstance(inp, FeedbackInput)
    assert inp.text == "Great transcription accuracy"


@pytest.mark.asyncio
async def test_validate_feedback_input_valid_rating():
    inp = await validate_feedback_input(
        text="",
        rating="up",
    )
    assert isinstance(inp, FeedbackInput)
    assert inp.rating == "up"


@pytest.mark.asyncio
async def test_validate_feedback_input_neither_raises_422():
    with pytest.raises(HTTPException) as exc_info:
        await validate_feedback_input(
            text="",
            rating=None,
        )
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_validate_feedback_input_invalid_rating():
    with pytest.raises(HTTPException) as exc_info:
        await validate_feedback_input(
            text="Nice",
            rating="not_a_valid_rating",
        )
    assert exc_info.value.status_code == 422
    json.dumps(exc_info.value.detail)


# ==============================================================================
# Output Coercion (Never-fail contract)
# ==============================================================================

def test_validate_extraction_output_coercion():
    raw_er = {
        "chief_complaint": ["Chest pain", "Shortness of breath"],
        "vital_signs": "BP 120/80",
        "unexpected_extra_key": "will be dropped or handled gracefully"
    }
    coerced = validate_extraction_output(raw_er, "er")
    assert isinstance(coerced, dict)
    # Lists should be coerced to string in output schema
    assert isinstance(coerced.get("chief_complaint"), str)
    assert "Chest pain" in coerced["chief_complaint"]


def test_validate_extraction_output_never_fail_on_invalid():
    raw_bad = {"invalid": 123}
    res = validate_extraction_output(raw_bad, "non_existent_department")
    assert isinstance(res, dict)
    assert "invalid" in res


def test_validate_diagnosis_output_coercion():
    raw_diag = {
        "icd10_codes": ["I10", "E11.9"],
        "clinical_reasoning": ["Patient has hypertension and diabetes"],
        "triage_priority": "urgent"
    }
    coerced = validate_diagnosis_output(raw_diag)
    assert isinstance(coerced, dict)
    assert isinstance(coerced.get("icd10_codes"), list)
    assert "I10" in coerced["icd10_codes"]
    assert isinstance(coerced.get("reasoning"), str)


def test_validate_diagnosis_output_never_fail():
    raw_bad = "completely non dict"
    res = validate_diagnosis_output(raw_bad)
    assert res == raw_bad
