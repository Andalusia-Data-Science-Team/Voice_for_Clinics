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
from model.diagnosis import constrain_diagnosis_to_registry
from model.patient_history import check_history_usage


def test_diagnosis_pipeline_maps_to_australian_codes():
    mock_llm_output = {
        "json_data": {
            "icd10_codes": [
                "A00.0 - Cholera due to Vibrio cholerae 01, biovar cholerae",
                "K04.0 - Pulpitis",
                "FAKE999.00 - Unsupported model suggestion",
            ],
            "reasoning": "Clinical assessment indicates multiple active conditions.",
            "history_considered": "No",
        },
        "summary": "Multi-system diagnosis review.",
    }

    with patch("model.llm_service.LLMService.recommend_diagnosis", return_value=mock_llm_output):
        with patch("core.config.Config.ai_foundry_key", return_value="test_key"):
            data, summary = DiagnosisRecommender.recommend(
                end_text="Patient presents with cholera and pulpitis.",
                department="opd",
            )
            # Only exact codes in this case's catalogue shortlist survive.
            assert len(data["icd10_codes"]) == 2
            assert any(c.startswith("A00.0") for c in data["icd10_codes"])
            assert any(c.startswith("K04.0") for c in data["icd10_codes"])
            assert data["icd10_unmatched"] == [
                "FAKE999.00 - Unsupported model suggestion"
            ]


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
        with patch("core.config.Config.ai_foundry_key", return_value="test_key"):
            data, summary = DiagnosisRecommender.recommend(
                end_text="Patient presents with pneumonia and chest symptoms.",
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


def test_recommendations_are_canonical_catalogue_entries_only():
    result = constrain_diagnosis_to_registry({
        "icd10_codes": [
            "J18.9 - model-invented description",
            "A00.0 - Valid code not in this case's shortlist",
            "FAKE999.00 - Hallucinated condition",
        ],
        "final_diagnosis": "Hallucinated condition",
        "reasoning": "Clinical reasoning text.",
    }, "opd", [("J18.9", "Pneumonia, unspecified")])

    assert result["icd10_codes"] == ["J18.9 - Pneumonia, unspecified"]
    assert result["final_diagnosis"] == "J18.9 - Pneumonia, unspecified"
    assert result["icd10_unmatched"] == [
        "A00.0 - Valid code not in this case's shortlist",
        "FAKE999.00 - Hallucinated condition",
    ]
    assert all(" - " in item for item in result["icd10_codes"])


def test_er_differentials_are_also_restricted_to_catalogue():
    result = constrain_diagnosis_to_registry({
        "icd10_codes": ["J18.9"],
        "differential_diagnoses": [
            "N23 - invented label",
            "Made-up diagnosis",
        ],
        "triage_priority": "ESI 3",
    }, "er", [
        ("J18.9", "Pneumonia, unspecified"),
        ("N23", "Unspecified renal colic"),
    ])

    assert result["differential_diagnoses"] == ["N23 - Unspecified renal colic"]
    assert result["differential_unmatched"] == ["Made-up diagnosis"]


def test_pneumonia_candidate_search_includes_unspecified_pneumonia():
    from model.icd10_registry import get_registry

    candidates = get_registry().search_diagnosis_candidates(
        "Patient presents with pneumonia and chest symptoms."
    )
    assert ("J18.9", "Pneumonia, unspecified") in candidates


def test_hybrid_retrieval_includes_relevant_catalogue_entries():
    from model.icd10_registry import get_registry

    registry = get_registry()
    cases = [
        (
            "Patient presents with pneumonia and chest symptoms.",
            "J18.9",
        ),
        (
            "Patient has type 2 diabetes mellitus.",
            "E11.9",
        ),
        (
            "Clinical presentation of cholera.",
            "A00.0",
        ),
        (
            "Severe tooth pain suggests pulpitis.",
            "K04.0",
        ),
    ]
    for note, expected_code in cases:
        candidates = registry.search_diagnosis_candidates(note)
        assert any(code == expected_code for code, _description in candidates)


def test_diagnosis_prompt_supplies_only_catalogue_shortlist():
    from utils.clinical_prompts import get_diagnosis_prompt

    prompt = get_diagnosis_prompt(
        "opd",
        "Patient presents with pneumonia.",
        catalogue_candidates=[("J18.9", "Pneumonia, unspecified")],
    )
    assert "- J18.9 - Pneumonia, unspecified" in prompt
    assert "select ONLY from these entries" in prompt
    assert "K35.8 - Acute appendicitis" not in prompt


if __name__ == "__main__":
    test_diagnosis_pipeline_maps_to_australian_codes()
    test_diagnosis_pipeline_surfaces_unmatched()
    test_history_check_still_works_with_validated_codes()
    print("All Phase 5.2 integration tests passed successfully!")
