"""
Unit Tests for Phase 2 (M3.2): Strict Pydantic Output Schemas & Validation Contracts.

Covers:
- M3.2 #1: Pydantic schemas for extraction and reasoning outputs (ER, OPD, Dental, Diagnosis)
- M3.2 #2: Invalid output rejected with structured reason, loc, and error_code
- Coercion, type safety, extra field resilience, and sub-1ms throughput
"""

import sys
import time
from pathlib import Path

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from validation.exceptions import ValidationErrorCode
from validation.schemas import (
    DentalExtractionSchema,
    DiagnosisReasoningSchema,
    ERExtractionSchema,
    OPDExtractionSchema,
    SchemaValidator,
)


def test_opd_extraction_schema_valid():
    """Verify valid OPD extraction data parses cleanly."""
    raw = {
        "chief_complaint": "Persistent cough and fever",
        "onset": "Gradual, started 5 days ago",
        "course": "Gradually worsening",
        "duration": "5 days",
        "comorbidities": "Type 2 diabetes",
        "medication_history": "Metformin 500mg twice daily",
        "past_medical_history": "Type 2 diabetes; seasonal allergies",
        "allergy_status": "No known drug allergies",
        "examination_finding": "Temp 38.2C, bilateral crackles on auscultation",
        "plan_of_care": "Start antibiotics, symptomatic treatment",
    }
    model, rejection = SchemaValidator.validate_extraction(raw, department="opd")
    assert rejection is None
    assert model is not None
    assert isinstance(model, OPDExtractionSchema)
    assert model.chief_complaint == "Persistent cough and fever"
    assert model.comorbidities == "Type 2 diabetes"
    assert model.plan_of_care == "Start antibiotics, symptomatic treatment"


def test_er_extraction_schema_valid():
    """Verify valid ER extraction parses primary survey and vitals."""
    raw = {
        "chief_complaint": "Acute RLQ pain",
        "onset": "6 hours ago",
        "airway": "Patent",
        "breathing": "Clear bilateral breath sounds",
        "circulation": "BP 130/85, HR 92",
        "disability": "GCS 15",
        "exposure": "Abdomen tender at McBurney's point",
        "examination": "Rebound tenderness present",
        "investigations": "Elevated WBC, US shows inflamed appendix",
        "plan_of_care": "Urgent appendectomy",
        "patient_disposition": "Admit to surgical ward",
    }
    model, rejection = SchemaValidator.validate_extraction(raw, department="er")
    assert rejection is None
    assert model is not None
    assert isinstance(model, ERExtractionSchema)
    assert model.airway == "Patent"
    assert model.circulation == "BP 130/85, HR 92"
    assert model.plan_of_care == "Urgent appendectomy"


def test_dental_extraction_schema_valid():
    """Verify valid Dental extraction parses dental-specific charting."""
    raw = {
        "chief_complaint": "Severe toothache on lower left molar",
        "tooth_number": "FDI 36",
        "subspecialty_focus": "Endodontics",
        "procedure_executed": "Root canal therapy step 1",
        "materials_inventory": "Sodium hypochlorite, Cavit",
        "consent_obtained": "Yes",
        "treatment_plan_status": "In-Progress",
    }
    model, rejection = SchemaValidator.validate_extraction(raw, department="dental")
    assert rejection is None
    assert model is not None
    assert isinstance(model, DentalExtractionSchema)
    assert model.tooth_number == "FDI 36"
    assert model.subspecialty_focus == "Endodontics"


def test_diagnosis_reasoning_schema_valid():
    """Verify valid Diagnosis output parses correctly."""
    raw = {
        "icd10_codes": ["J18.9 - Pneumonia, unspecified organism", "J20.9 - Acute bronchitis"],
        "reasoning": "Fever and crackles favour pneumonia.",
        "history_considered": "No",
        "history_notes": "",
        "differential_diagnoses": ["J20.9 - Acute bronchitis"],
        "triage_priority": "Category 3 - Urgent",
    }
    model, rejection = SchemaValidator.validate_diagnosis(raw)
    assert rejection is None
    assert model is not None
    assert isinstance(model, DiagnosisReasoningSchema)
    assert len(model.icd10_codes) == 2
    assert model.triage_priority == "Category 3 - Urgent"


def test_invalid_extraction_rejected_with_reason():
    """Verify non-dictionary or corrupted extraction is rejected with structured reason."""
    # 1. Non-dict input (e.g. list, string, int)
    model, rejection = SchemaValidator.validate_extraction("Invalid string instead of JSON dict", department="opd")
    assert model is None
    assert rejection is not None
    assert rejection.status == "rejected"
    assert rejection.error_code == ValidationErrorCode.UNEXPECTED_DATA_SHAPE.value
    assert "must be a dictionary" in rejection.reason
    assert rejection.loc == ["root"]

    # 2. Corrupted input (None)
    model_none, rejection_none = SchemaValidator.validate_extraction(None, department="er")
    assert model_none is None
    assert rejection_none is not None
    assert rejection_none.status == "rejected"


def test_invalid_diagnosis_rejected_with_reason():
    """Verify non-dictionary or corrupted diagnosis output is rejected with reason."""
    model, rejection = SchemaValidator.validate_diagnosis(["Array instead of dict"])
    assert model is None
    assert rejection is not None
    assert rejection.status == "rejected"
    assert rejection.error_code == ValidationErrorCode.UNEXPECTED_DATA_SHAPE.value
    assert "must be a dictionary" in rejection.reason


def test_coercion_and_resilience():
    """Verify list-of-strings coercion into clean text fields without failing."""
    raw = {
        "chief_complaint": ["Fever", "Cough", "Headache"],  # model returned a list
        "plan_of_care": 12345,                              # model returned an int
        "allergy_status": None,                             # model returned None
    }
    model, rejection = SchemaValidator.validate_extraction(raw, department="opd")
    assert rejection is None
    assert model is not None
    assert "Fever, Cough, Headache" in model.chief_complaint
    assert model.plan_of_care == "12345"
    assert model.allergy_status == ""


def test_validation_performance():
    """Verify schema validation executes in sub-millisecond (< 1ms per record)."""
    raw = {
        "chief_complaint": "Chest discomfort",
        "onset": "1 hour ago",
        "airway": "Patent",
        "breathing": "Symmetrical",
        "circulation": "BP 140/90",
        "disability": "Alert",
        "exposure": "No trauma",
        "plan_of_care": "ECG, Troponin, Aspirin",
    }
    # Warm-up
    for _ in range(50):
        SchemaValidator.validate_extraction(raw, department="er")

    t0 = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        SchemaValidator.validate_extraction(raw, department="er")
    total_ms = (time.perf_counter() - t0) * 1000
    avg_ms = total_ms / iterations

    assert avg_ms < 1.0, f"Average validation time {avg_ms:.4f}ms exceeded 1ms budget"


if __name__ == "__main__":
    test_opd_extraction_schema_valid()
    test_er_extraction_schema_valid()
    test_dental_extraction_schema_valid()
    test_diagnosis_reasoning_schema_valid()
    test_invalid_extraction_rejected_with_reason()
    test_invalid_diagnosis_rejected_with_reason()
    test_coercion_and_resilience()
    test_validation_performance()
    print("All Phase 2 (M3.2) schema tests passed successfully!")
