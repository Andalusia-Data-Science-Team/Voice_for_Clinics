from utils.red_flags import detect_red_flags
from schemas.output_schemas import PipelineSessionOutput


def test_detects_english_red_flag_case_insensitively():
    result = detect_red_flags("Patient reports SHORTNESS OF BREATH.")

    assert result["requires_clinician_review"] is True
    assert result["action"] == "clinician_review"
    assert result["matched_signals"][0]["id"] == "breathing_difficulty"


def test_detects_arabic_red_flag_with_diacritics_and_alef_variants():
    result = detect_red_flags("المريض لديه ضِيقُ التَّنَفُّس")

    assert result["requires_clinician_review"] is True
    assert result["matched_signals"][0]["id"] == "breathing_difficulty"


def test_phrase_boundary_avoids_partial_english_word_matches():
    result = detect_red_flags("The note says chest pains were not discussed.")

    assert result["requires_clinician_review"] is False


def test_negated_english_phrase_is_not_routed_as_a_positive_match():
    result = detect_red_flags("Patient denies chest pain.")

    assert result["requires_clinician_review"] is False
    assert result["action"] == "negated_only"
    assert result["matched_signals"][0]["status"] == "negated"


def test_negation_does_not_cancel_a_later_contrastive_positive_finding():
    result = detect_red_flags("Denies chest pain, but has difficulty breathing.")

    assert result["requires_clinician_review"] is True
    statuses = {item["id"]: item["status"] for item in result["matched_signals"]}
    assert statuses["chest_pain"] == "negated"
    assert statuses["breathing_difficulty"] == "affirmed"


def test_uncertain_phrase_is_routed_for_clinician_review():
    result = detect_red_flags("Possible chest pain.")

    assert result["requires_clinician_review"] is True
    assert result["matched_signals"][0]["status"] == "uncertain"


def test_negated_arabic_phrase_is_not_routed_as_a_positive_match():
    result = detect_red_flags("المريض لا يعاني من ضيق التنفس")

    assert result["requires_clinician_review"] is False
    assert result["matched_signals"][0]["status"] == "negated"


def test_no_match_is_not_presented_as_excluding_an_urgent_condition():
    result = detect_red_flags("Routine follow-up for a skin rash.")

    assert result["requires_clinician_review"] is False
    assert result["action"] == "none_detected"
    assert "does not rule out an urgent condition" in result["notice"]


def test_routing_signal_is_retained_in_persisted_session_schema():
    routing_signal = detect_red_flags("Patient reports chest pain.")

    session = PipelineSessionOutput(
        department="opd",
        language="en",
        mode="doctor",
        routing_signal=routing_signal,
    )

    assert session.model_dump()["routing_signal"] == routing_signal
