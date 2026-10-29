from model.patient_history import validate_history_citations


VISITS = [
    {"visit_date": "2025-02-14", "icd10_diagnoses": ["J18.9"]},
    {"visit_date": "2024-11-03", "icd10_diagnoses": ["J20.9"]},
]


def test_history_citation_dates_are_verified_against_visits():
    result = validate_history_citations(
        {
            "history_considered": "Yes",
            "history_notes": "Prior pneumonia J18.9 on 2025-02-14 informed the recommendation.",
            "reasoning": "The prior pneumonia on 2025-02-14 supports the recommendation.",
        },
        VISITS,
    )

    assert result["history_cited_visit_dates"] == ["2025-02-14"]
    assert result["history_citation_missing"] is False


def test_history_citation_is_flagged_when_visit_date_is_missing():
    result = validate_history_citations(
        {
            "history_considered": "Yes",
            "history_notes": "Prior pneumonia informed the recommendation.",
            "reasoning": "Prior pneumonia supports the recommendation.",
        },
        VISITS,
    )

    assert result["history_cited_visit_dates"] == []
    assert result["history_citation_missing"] is True


def test_history_citation_is_flagged_when_date_is_not_a_supplied_visit():
    result = validate_history_citations(
        {
            "history_considered": "Yes",
            "history_notes": "Prior pneumonia on 2025-01-01 informed the recommendation.",
            "reasoning": "Prior pneumonia on 2025-01-01 supports the recommendation.",
        },
        VISITS,
    )

    assert result["history_cited_visit_dates"] == []
    assert result["history_citation_missing"] is True


def test_history_citation_is_flagged_when_date_is_omitted_from_reasoning():
    result = validate_history_citations(
        {
            "history_considered": "Yes",
            "history_notes": "Prior pneumonia J18.9 on 2025-02-14 informed the recommendation.",
            "reasoning": "Prior pneumonia supports the recommendation.",
        },
        VISITS,
    )

    assert result["history_cited_visit_dates"] == ["2025-02-14"]
    assert result["history_citation_missing"] is True


def test_no_history_impact_does_not_require_citation():
    result = validate_history_citations(
        {
            "history_considered": "No",
            "history_notes": "",
        },
        VISITS,
    )

    assert result["history_cited_visit_dates"] == []
    assert result["history_citation_missing"] is False
