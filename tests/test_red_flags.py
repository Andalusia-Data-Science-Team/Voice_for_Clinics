import json

from utils.red_flags import LEXICON_PATH, detect_red_flags
from schemas.output_schemas import PipelineSessionOutput


def test_detects_english_red_flag_case_insensitively():
    result = detect_red_flags("Patient reports SHORTNESS OF BREATH.")

    assert result["requires_clinician_review"] is True
    assert result["action"] == "clinician_review"
    assert result["matched_signals"][0]["id"] == "shortness_of_breath"


def test_detects_arabic_red_flag_with_diacritics_and_alef_variants():
    result = detect_red_flags("المريض لديه ضِيقُ التَّنَفُّس")

    assert result["requires_clinician_review"] is True
    assert result["matched_signals"][0]["id"] == "shortness_of_breath"


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
    assert statuses["chest_pain_cardiac_features"] == "negated"
    assert statuses["shortness_of_breath"] == "affirmed"


def test_uncertain_phrase_is_routed_for_clinician_review():
    result = detect_red_flags("Possible chest pain.")

    assert result["requires_clinician_review"] is True
    assert result["matched_signals"][0]["status"] == "uncertain"


def test_negated_arabic_phrase_is_not_routed_as_a_positive_match():
    result = detect_red_flags("المريض لا يعاني من ضيق التنفس")

    assert result["requires_clinician_review"] is False
    assert result["matched_signals"][0]["status"] == "negated"


def test_no_match_is_not_presented_as_excluding_an_urgent_condition():
    result = detect_red_flags("Routine follow-up, patient feels well.")

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


def test_every_lexicon_term_is_detected_from_its_own_phrases():
    lexicon = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))

    assert len(lexicon) == 169
    owners: dict[str, str] = {}
    for entry in lexicon:
        for phrase in entry["phrases"]["en"]:
            assert owners.setdefault(phrase, entry["id"]) == entry["id"], phrase
            result = detect_red_flags(f"Patient presents with {phrase}.")
            assert entry["id"] in {m["id"] for m in result["matched_signals"]}, (
                entry["id"], phrase,
            )
            assert result["requires_clinician_review"] is True


def test_matches_carry_triage_acuity_and_population():
    result = detect_red_flags("Child has stridor and a mild headache.")

    by_id = {m["id"]: m for m in result["matched_signals"]}
    assert by_id["stridor"]["acuity"] == 3
    assert by_id["stridor"]["population"] == "pediatric"
    assert by_id["headache"]["acuity"] == 5
    assert result["highest_acuity"] == 3
    assert [m["acuity"] for m in result["matched_signals"]] == [3, 5]


def test_longest_phrase_wins_when_phrases_overlap():
    result = detect_red_flags("Reports non-cardiac chest pain on deep breaths.")

    assert [m["id"] for m in result["matched_signals"]] == [
        "chest_pain_non_cardiac_features"
    ]


def test_negated_only_matches_have_no_highest_acuity():
    result = detect_red_flags("Denies fever.")

    assert result["highest_acuity"] is None
    assert result["action"] == "negated_only"


def test_cardiac_arrest_is_acuity_one():
    result = detect_red_flags("Found in cardiac arrest at home.")

    assert result["highest_acuity"] == 1


def test_user_provided_red_flag_terms_trigger_doctor_alerts():
    phrases = [
        "Leg swelling / edema",
        "Palpitations / irregular heart beat",
        "Depression / suicidal / deliberate self-harm",
        "Discharge, ear",
        "Foreign body, vagina",
        "Joint(s) swelling",
        "Foreign body, skin",
        "Isolated abdominal trauma – blunt",
        "Concern for the patient’s welfare",
    ]

    for phrase in phrases:
        result = detect_red_flags(f"Patient reports {phrase}.")
        assert result["requires_clinician_review"] is True, phrase
