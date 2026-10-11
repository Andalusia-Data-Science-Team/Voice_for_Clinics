"""Phase 5.2 Integration Tests for Diagnosis Pipeline and History Checking.

Covers:
- test_diagnosis_pipeline_maps_to_australian_codes()
- test_diagnosis_pipeline_surfaces_unmatched()
- test_history_check_still_works_with_validated_codes()
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure src is in python path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.diagnosis import DiagnosisRecommender
from model.patient_history import check_history_usage


def test_diagnosis_pipeline_maps_to_australian_codes():
    mock_llm_output = {
        "json_data": {
            "icd10_codes": [
                "A00.0 - Cholera due to Vibrio cholerae 01, biovar cholerae",
                "J18.99 - Pneumonia condition",  # progressive fallback to J18.9
                "E11.9 - Type 2 diabetes mellitus without complications",
            ],
            "reasoning": "Clinical assessment indicates multiple active conditions.",
            "history_considered": "No",
        },
        "summary": "Multi-system diagnosis review.",
    }

    with patch("model.llm_service.LLMService.recommend_diagnosis", return_value=mock_llm_output):
        with patch("core.config.Config.openrouter_api_key", return_value="test_key"):
            data, summary = DiagnosisRecommender.recommend(
                end_text="Patient presents with fever, productive cough, and thirst.",
                department="opd",
            )
            # All 3 codes must be successfully mapped to official Australian codes
            assert len(data["icd10_codes"]) == 3
            assert any(c.startswith("A00.0") for c in data["icd10_codes"])
            assert any(c.startswith("J18.9") for c in data["icd10_codes"])
            assert any(c.startswith("E11.9") for c in data["icd10_codes"])
            assert len(data.get("icd10_unmatched", [])) == 0


def test_diagnosis_pipeline_surfaces_unmatched():
    mock_llm_output = {
        "json_data": {
            "icd10_codes": [
                "J18.9 - Pneumonia, unspecified",
                "FAKE999.00 - Hallucinated condition name",
                "XYZ.123 - Non-existent medical category",
            ],
            "reasoning": "Partial clinical match.",
            "history_considered": "No",
        },
        "summary": "Evaluated with outliers.",
    }

    with patch("model.llm_service.LLMService.recommend_diagnosis", return_value=mock_llm_output):
        with patch("core.config.Config.openrouter_api_key", return_value="test_key"):
            data, summary = DiagnosisRecommender.recommend(
                end_text="Patient presents with chest symptoms.",
                department="opd",
            )
            # Only J18.9 is valid
            assert len(data["icd10_codes"]) == 1
            assert data["icd10_codes"][0].startswith("J18.9")

            # Both invalid codes must be isolated in icd10_unmatched
            unmatched = data.get("icd10_unmatched", [])
            assert len(unmatched) == 2
            assert "FAKE999.00 - Hallucinated condition name" in unmatched
            assert "XYZ.123 - Non-existent medical category" in unmatched


def test_history_check_still_works_with_validated_codes():
    # Past visit history from hospital database
    mock_visits = [
        {
            "visit_date": "2025-05-10",
            "icd10_diagnoses": ["E11.9 - Type 2 diabetes mellitus without complications"],
        }
    ]

    # Validated codes returned by DiagnosisRecommender
    extracted_features = {
        "icd10_codes": ["E11.01 - Type 2 diabetes with hyperosmolarity", "J18.9 - Pneumonia"],
        "history_considered": "Yes",
        "history_notes": "Reinforced diabetic follow-up.",
    }

    result = check_history_usage(mock_visits, extracted_features)
    assert result["history_provided"] is True
    assert result["overlap_found"] is True
    assert "E11" in result["overlap_families"]
    assert result["flag"] is None


if __name__ == "__main__":
    test_diagnosis_pipeline_maps_to_australian_codes()
    test_diagnosis_pipeline_surfaces_unmatched()
    test_history_check_still_works_with_validated_codes()
    print("All Phase 5.2 integration tests passed successfully!")
