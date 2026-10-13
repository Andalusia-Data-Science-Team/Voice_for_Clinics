"""
tests/test_schemas.py
=====================
Phase 1 — Unit tests for all Pydantic input/output schemas.

Coverage matrix
---------------
Task 1.1 __init__.py       — public re-export smoke-tests
Task 1.2 input_schemas.py  — AnalyzeInput, RefineFieldInput,
                             WebSocketInitInput, FeedbackInput
Task 1.3 output_schemas.py — ERExtractionOutput, OPDExtractionOutput,
                             DentalExtractionOutput, DiagnosisOutput,
                             UsageMeta, QuestionItem, PipelineSessionOutput
Task 1.4 validators.py     — sanitize_medical_text, coerce_to_str,
                             coerce_to_str_list, normalize_patient_id,
                             clamp_float, require_non_blank

Each section tests:
  ✓ Valid baseline — canonical happy-path input
  ✓ Edge cases    — boundary values, optional fields absent/present
  ✗ Invalid cases — expected ValidationError or ValueError
  ✓ Coercion      — int/list/None → str where specified
  ✓ Performance   — sub-ms per-validation average for hot schemas
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

# Ensure src/ is on the path regardless of how pytest is invoked.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Imports under test ────────────────────────────────────────────────────────
from schemas import (
    AnalyzeInput,
    BusinessUnit,
    DentalExtractionOutput,
    Department,
    DiagnosisOutput,
    ERExtractionOutput,
    EXTRACTION_SCHEMAS,
    FeedbackInput,
    Language,
    MAX_FIELD_TEXT_LEN,
    MAX_NAME_LEN,
    MAX_PATIENT_ID_LEN,
    MAX_SESSION_ID_LEN,
    MAX_TEXT_LEN,
    Mode,
    OPDExtractionOutput,
    PipelineSessionOutput,
    QuestionItem,
    RefineFieldInput,
    UsageMeta,
    WebSocketInitInput,
    clamp_float,
    coerce_to_str,
    coerce_to_str_list,
    get_extraction_output_schema,
    normalize_patient_id,
    require_non_blank,
    sanitize_medical_text,
)


# ══════════════════════════════════════════════════════════════════════════════
# §1  __init__.py — public re-export smoke-tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPublicAPI:
    """Verify everything promised in __init__.py is actually importable."""

    def test_all_input_schemas_importable(self):
        assert AnalyzeInput is not None
        assert RefineFieldInput is not None
        assert WebSocketInitInput is not None
        assert FeedbackInput is not None

    def test_all_enums_importable(self):
        assert Language.arabic.value == "ar"
        assert Language.english.value == "en"
        assert Mode.doctor.value == "doctor"
        assert Mode.conversation.value == "conversation"
        assert Department.er.value == "er"
        assert Department.opd.value == "opd"
        assert Department.dental.value == "dental"
        assert BusinessUnit.AHJ.value == "AHJ"
        assert BusinessUnit.AMH.value == "AMH"
        assert BusinessUnit.ANC.value == "ANC"

    def test_all_output_schemas_importable(self):
        assert ERExtractionOutput is not None
        assert OPDExtractionOutput is not None
        assert DentalExtractionOutput is not None
        assert DiagnosisOutput is not None
        assert UsageMeta is not None
        assert QuestionItem is not None
        assert PipelineSessionOutput is not None

    def test_extraction_schemas_registry_complete(self):
        assert set(EXTRACTION_SCHEMAS.keys()) == {"er", "opd", "dental"}

    def test_constants_are_positive_ints(self):
        for const in (MAX_TEXT_LEN, MAX_FIELD_TEXT_LEN, MAX_PATIENT_ID_LEN,
                      MAX_SESSION_ID_LEN, MAX_NAME_LEN):
            assert isinstance(const, int) and const > 0


# ══════════════════════════════════════════════════════════════════════════════
# §2  validators.py
# ══════════════════════════════════════════════════════════════════════════════

class TestSanitizeMedicalText:
    """sanitize_medical_text — Unicode-safe whitespace normalizer."""

    def test_strips_leading_trailing_whitespace(self):
        assert sanitize_medical_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert sanitize_medical_text("hello   world") == "hello world"

    def test_handles_tabs_and_newlines(self):
        assert sanitize_medical_text("hello\t\nworld") == "hello world"

    def test_arabic_text_preserved(self):
        arabic = "  المريض يعاني من ألم  "
        result = sanitize_medical_text(arabic)
        assert result == "المريض يعاني من ألم"

    def test_mixed_arabic_english(self):
        mixed = "Patient (مريض) has   fever"
        result = sanitize_medical_text(mixed)
        assert result == "Patient (مريض) has fever"

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="non-whitespace"):
            sanitize_medical_text("")

    def test_raises_on_whitespace_only(self):
        with pytest.raises(ValueError, match="non-whitespace"):
            sanitize_medical_text("   \t\n  ")

    def test_field_name_in_error_message(self):
        with pytest.raises(ValueError, match="'transcript'"):
            sanitize_medical_text("", field_name="transcript")


class TestCoerceToStr:
    """coerce_to_str — any-to-str converter."""

    def test_str_passthrough(self):
        assert coerce_to_str("hello") == "hello"

    def test_none_returns_empty(self):
        assert coerce_to_str(None) == ""

    def test_none_custom_default(self):
        assert coerce_to_str(None, none_value="N/A") == "N/A"

    def test_int_coerced(self):
        assert coerce_to_str(42) == "42"

    def test_list_joined(self):
        assert coerce_to_str(["Fever", "Cough"]) == "Fever, Cough"

    def test_list_with_none_skipped(self):
        assert coerce_to_str(["Fever", None, "Cough"]) == "Fever, Cough"

    def test_dict_values_joined(self):
        result = coerce_to_str({"a": "Fever", "b": "Cough"})
        assert "Fever" in result and "Cough" in result

    def test_empty_list_returns_empty(self):
        assert coerce_to_str([]) == ""


class TestCoerceToStrList:
    """coerce_to_str_list — any-to-List[str] converter."""

    def test_none_returns_empty_list(self):
        assert coerce_to_str_list(None) == []

    def test_str_wrapped_in_list(self):
        assert coerce_to_str_list("J18.9") == ["J18.9"]

    def test_blank_str_returns_empty_list(self):
        assert coerce_to_str_list("   ") == []

    def test_list_cleaned(self):
        result = coerce_to_str_list(["J18.9", "  J20.9  ", None])
        assert result == ["J18.9", "J20.9"]

    def test_dict_values(self):
        result = coerce_to_str_list({"a": "J18.9", "b": "J20.9"})
        assert set(result) == {"J18.9", "J20.9"}

    def test_empty_list(self):
        assert coerce_to_str_list([]) == []


class TestNormalizePatientId:
    """normalize_patient_id — strips and uppercases patient code."""

    def test_none_passthrough(self):
        assert normalize_patient_id(None) is None

    def test_lowercased_to_upper(self):
        assert normalize_patient_id("abc-123") == "ABC-123"

    def test_strips_whitespace(self):
        assert normalize_patient_id("  P001  ") == "P001"

    def test_blank_returns_none(self):
        assert normalize_patient_id("   ") is None


class TestClampFloat:
    """clamp_float — bounds clamping for optional floats."""

    def test_none_passthrough(self):
        assert clamp_float(None) is None

    def test_within_range(self):
        assert clamp_float(5.0, lo=0.0, hi=10.0) == 5.0

    def test_below_lo_clamped(self):
        assert clamp_float(-1.0, lo=0.0, hi=10.0) == 0.0

    def test_above_hi_clamped(self):
        assert clamp_float(20.0, lo=0.0, hi=10.0) == 10.0

    def test_default_lo_zero(self):
        assert clamp_float(-5.0) == 0.0


class TestRequireNonBlank:
    """require_non_blank — asserts non-empty after strip."""

    def test_valid_string(self):
        assert require_non_blank("hello") == "hello"

    def test_strips_whitespace(self):
        assert require_non_blank("  hello  ") == "hello"

    def test_none_raises(self):
        with pytest.raises(ValueError, match="must not be blank"):
            require_non_blank(None)  # type: ignore[arg-type]

    def test_blank_raises(self):
        with pytest.raises(ValueError):
            require_non_blank("   ")


# ══════════════════════════════════════════════════════════════════════════════
# §3  AnalyzeInput
# ══════════════════════════════════════════════════════════════════════════════

_VALID_ANALYZE: dict[str, Any] = {
    "text": "Patient presents with chest pain for 2 hours.",
    "language": "en",
    "mode": "doctor",
    "department": "opd",
    "specialty": None,
    "use_history": False,
    "patient_id": None,
    "business_unit": None,
    "clinician_name": None,
    "transcription_sec": None,
    "session_id": None,
}


class TestAnalyzeInput:
    """AnalyzeInput — POST /audio/analyze request schema."""

    # ── Valid baseline ────────────────────────────────────────────────────────

    def test_valid_baseline(self):
        m = AnalyzeInput(**_VALID_ANALYZE)
        assert m.text == "Patient presents with chest pain for 2 hours."
        assert m.language == Language.english
        assert m.mode == Mode.doctor
        assert m.department == Department.opd

    def test_arabic_language_enum(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "language": "ar"})
        assert m.language == Language.arabic

    def test_er_department(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "department": "er"})
        assert m.department == Department.er

    def test_conversation_mode(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "mode": "conversation"})
        assert m.mode == Mode.conversation

    def test_use_history_with_patient_and_bu(self):
        m = AnalyzeInput(**{
            **_VALID_ANALYZE,
            "use_history": True,
            "patient_id": "P001",
            "business_unit": "AHJ",
        })
        assert m.use_history is True
        assert m.patient_id == "P001"
        assert m.business_unit == BusinessUnit.AHJ

    def test_session_id_accepted(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "session_id": "sess-abc-123"})
        assert m.session_id == "sess-abc-123"

    def test_transcription_sec_accepted(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "transcription_sec": 45.5})
        assert m.transcription_sec == 45.5

    def test_text_whitespace_is_normalized(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "text": "  Hello   world  "})
        assert m.text == "Hello world"

    def test_specialty_whitespace_stripped(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "specialty": "  Cardiology  "})
        assert m.specialty == "Cardiology"

    def test_blank_specialty_becomes_none(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "specialty": "   "})
        assert m.specialty is None

    def test_clinician_name_stripped(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "clinician_name": "  Dr. Ahmed  "})
        assert m.clinician_name == "Dr. Ahmed"

    def test_blank_clinician_name_becomes_none(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "clinician_name": "   "})
        assert m.clinician_name is None

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_max_length_text_accepted(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "text": "a" * MAX_TEXT_LEN})
        assert len(m.text) == MAX_TEXT_LEN

    def test_transcription_sec_zero_accepted(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "transcription_sec": 0.0})
        assert m.transcription_sec == 0.0

    def test_transcription_sec_max_accepted(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "transcription_sec": 3600.0})
        assert m.transcription_sec == 3600.0

    def test_patient_id_pattern_valid(self):
        m = AnalyzeInput(**{
            **_VALID_ANALYZE,
            "use_history": True,
            "patient_id": "PAT-0001-A",
            "business_unit": "AMH",
        })
        assert m.patient_id == "PAT-0001-A"

    def test_dental_department(self):
        m = AnalyzeInput(**{**_VALID_ANALYZE, "department": "dental"})
        assert m.department == Department.dental

    # ── Invalid cases ─────────────────────────────────────────────────────────

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "text": ""})

    def test_whitespace_only_text_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "text": "   "})

    def test_text_too_long_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "text": "x" * (MAX_TEXT_LEN + 1)})

    def test_invalid_language_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "language": "fr"})

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "mode": "nurse"})

    def test_invalid_department_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "department": "radiology"})

    def test_invalid_business_unit_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "business_unit": "UNKNOWN"})

    def test_use_history_without_patient_id_rejected(self):
        with pytest.raises(ValidationError, match="use_history requires"):
            AnalyzeInput(**{
                **_VALID_ANALYZE,
                "use_history": True,
                "patient_id": None,
                "business_unit": "AHJ",
            })

    def test_use_history_without_business_unit_rejected(self):
        with pytest.raises(ValidationError, match="use_history requires"):
            AnalyzeInput(**{
                **_VALID_ANALYZE,
                "use_history": True,
                "patient_id": "P001",
                "business_unit": None,
            })

    def test_invalid_patient_id_pattern_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{
                **_VALID_ANALYZE,
                "use_history": True,
                "patient_id": "P@001!",
                "business_unit": "AHJ",
            })

    def test_transcription_sec_negative_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "transcription_sec": -1.0})

    def test_transcription_sec_exceeds_hour_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "transcription_sec": 3601.0})

    def test_session_id_too_long_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "session_id": "x" * (MAX_SESSION_ID_LEN + 1)})

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeInput(**{**_VALID_ANALYZE, "unknown_field": "bad"})

    def test_arabic_text_valid(self):
        arabic = "المريض يشكو من ألم في الصدر منذ ساعتين"
        m = AnalyzeInput(**{**_VALID_ANALYZE, "text": arabic})
        assert m.text == arabic


# ══════════════════════════════════════════════════════════════════════════════
# §4  RefineFieldInput
# ══════════════════════════════════════════════════════════════════════════════

class TestRefineFieldInput:
    """RefineFieldInput — POST /audio/refine-field."""

    def test_valid_baseline(self):
        m = RefineFieldInput(text="Chest pain", language="en")
        assert m.text == "Chest pain"
        assert m.language == Language.english

    def test_arabic_text_accepted(self):
        m = RefineFieldInput(text="ألم في الصدر", language="ar")
        assert m.language == Language.arabic

    def test_specialty_accepted(self):
        m = RefineFieldInput(text="Cough", language="en", specialty="Pulmonology")
        assert m.specialty == "Pulmonology"

    def test_blank_specialty_normalised_to_none(self):
        m = RefineFieldInput(text="Cough", language="en", specialty="  ")
        assert m.specialty is None

    def test_text_whitespace_normalised(self):
        m = RefineFieldInput(text="  cough  fever  ", language="en")
        assert m.text == "cough fever"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            RefineFieldInput(text="", language="en")

    def test_whitespace_only_text_rejected(self):
        with pytest.raises(ValidationError):
            RefineFieldInput(text="   ", language="en")

    def test_text_too_long_rejected(self):
        with pytest.raises(ValidationError):
            RefineFieldInput(text="x" * (MAX_FIELD_TEXT_LEN + 1), language="en")

    def test_max_length_text_accepted(self):
        m = RefineFieldInput(text="a" * MAX_FIELD_TEXT_LEN, language="en")
        assert len(m.text) == MAX_FIELD_TEXT_LEN

    def test_invalid_language_rejected(self):
        with pytest.raises(ValidationError):
            RefineFieldInput(text="hello", language="fr")


# ══════════════════════════════════════════════════════════════════════════════
# §5  WebSocketInitInput
# ══════════════════════════════════════════════════════════════════════════════

class TestWebSocketInitInput:
    """WebSocketInitInput — WS /ws/transcribe init message."""

    def test_valid_baseline(self):
        m = WebSocketInitInput()
        assert m.sampleRate == 16_000
        assert m.language == "ar"
        assert m.hotwords is None

    def test_english_language(self):
        m = WebSocketInitInput(language="en")
        assert m.language == "en"

    def test_custom_sample_rate(self):
        m = WebSocketInitInput(sampleRate=48_000)
        assert m.sampleRate == 48_000

    def test_hotwords_accepted(self):
        m = WebSocketInitInput(hotwords="hospital, clinic, department")
        assert m.hotwords == "hospital, clinic, department"

    def test_blank_hotwords_normalised_to_none(self):
        m = WebSocketInitInput(hotwords="   ")
        assert m.hotwords is None

    def test_sample_rate_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketInitInput(sampleRate=7_999)

    def test_sample_rate_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketInitInput(sampleRate=48_001)

    def test_invalid_language_rejected(self):
        with pytest.raises(ValidationError):
            WebSocketInitInput(language="de")

    def test_sample_rate_minimum_boundary(self):
        m = WebSocketInitInput(sampleRate=8_000)
        assert m.sampleRate == 8_000

    def test_sample_rate_maximum_boundary(self):
        m = WebSocketInitInput(sampleRate=48_000)
        assert m.sampleRate == 48_000


# ══════════════════════════════════════════════════════════════════════════════
# §6  FeedbackInput
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedbackInput:
    """FeedbackInput — POST /feedback."""

    def test_text_only(self):
        m = FeedbackInput(text="Great tool!", rating=None)
        assert m.text == "Great tool!"
        assert m.rating is None

    def test_rating_up(self):
        m = FeedbackInput(rating="up", text="")
        assert m.rating == "up"

    def test_rating_down(self):
        m = FeedbackInput(rating="down", text="")
        assert m.rating == "down"

    def test_both_text_and_rating(self):
        m = FeedbackInput(rating="up", text="Excellent")
        assert m.rating == "up"
        assert m.text == "Excellent"

    def test_empty_rating_normalised_to_none(self):
        m = FeedbackInput(text="something", rating="")
        assert m.rating is None

    def test_name_accepted(self):
        m = FeedbackInput(text="good", name="Dr. Ahmed")
        assert m.name == "Dr. Ahmed"

    def test_blank_name_normalised_to_none(self):
        m = FeedbackInput(text="good", name="  ")
        assert m.name is None

    def test_department_and_language(self):
        m = FeedbackInput(text="good", department="er", language="ar")
        assert m.department == "er"
        assert m.language == "ar"

    def test_session_id_accepted(self):
        m = FeedbackInput(text="good", session_id="abc-123")
        assert m.session_id == "abc-123"

    def test_no_text_no_rating_rejected(self):
        with pytest.raises(ValidationError, match="thumbs rating"):
            FeedbackInput(text="", rating=None)

    def test_whitespace_only_text_and_no_rating_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackInput(text="   ", rating=None)

    def test_invalid_rating_rejected(self):
        with pytest.raises(ValidationError, match='rating must be "up" or "down"'):
            FeedbackInput(text="something", rating="neutral")

    def test_session_id_too_long_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackInput(text="good", session_id="x" * (MAX_SESSION_ID_LEN + 1))

    def test_name_too_long_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackInput(text="good", name="x" * (MAX_NAME_LEN + 1))


# ══════════════════════════════════════════════════════════════════════════════
# §7  ERExtractionOutput
# ══════════════════════════════════════════════════════════════════════════════

class TestERExtractionOutput:
    """ERExtractionOutput — ER department extraction output schema."""

    def test_valid_baseline(self):
        data = {
            "chief_complaint": "Acute RLQ pain",
            "onset": "6 hours ago",
            "airway": "Patent",
            "breathing": "Clear bilateral breath sounds",
            "circulation": "BP 130/85, HR 92",
            "disability": "GCS 15",
            "exposure": "Abdomen tender at McBurney's point",
            "examination": "Rebound tenderness present",
            "investigations": "Elevated WBC; US shows inflamed appendix",
            "plan_of_care": "Urgent appendectomy",
            "patient_disposition": "Admit to surgical ward",
        }
        m = ERExtractionOutput.model_validate(data)
        assert m.chief_complaint == "Acute RLQ pain"
        assert m.airway == "Patent"
        assert m.plan_of_care == "Urgent appendectomy"
        # Unset fields should be None
        assert m.sick_leave is None
        assert m.discharge_instructions is None

    def test_all_fields_none_by_default(self):
        m = ERExtractionOutput()
        assert m.chief_complaint is None
        assert m.sick_leave is None
        assert m.discharge_instructions is None
        # triage_priority belongs to DiagnosisOutput, not ERExtractionOutput
        assert not hasattr(m, "triage_priority")

    def test_extra_keys_ignored(self):
        m = ERExtractionOutput.model_validate({"chief_complaint": "Pain", "unknown_key": "ignored"})
        assert m.chief_complaint == "Pain"
        assert not hasattr(m, "unknown_key")

    def test_list_coerced_to_str(self):
        m = ERExtractionOutput.model_validate({"chief_complaint": ["Pain", "Fever"]})
        assert m.chief_complaint == "Pain, Fever"

    def test_int_coerced_to_str(self):
        m = ERExtractionOutput.model_validate({"onset": 3})
        assert m.onset == "3"

    def test_none_preserved(self):
        m = ERExtractionOutput.model_validate({"airway": None})
        assert m.airway is None

    def test_all_er_fields_present(self):
        """Every key in DEPARTMENT_FIELDS['er'] must be a field on the schema."""
        from model.department_fields import DEPARTMENT_FIELDS
        er_keys = {f["key"] for f in DEPARTMENT_FIELDS["er"]}
        schema_fields = set(ERExtractionOutput.model_fields.keys())
        missing = er_keys - schema_fields
        assert not missing, f"Missing ER fields in schema: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# §8  OPDExtractionOutput
# ══════════════════════════════════════════════════════════════════════════════

class TestOPDExtractionOutput:
    """OPDExtractionOutput — OPD department extraction output schema."""

    def test_valid_baseline(self):
        data = {
            "chief_complaint": "Persistent cough and fever",
            "onset": "5 days ago",
            "duration": "5 days",
            "comorbidities": "Type 2 diabetes",
            "plan_of_care": "Start antibiotics",
        }
        m = OPDExtractionOutput.model_validate(data)
        assert m.chief_complaint == "Persistent cough and fever"
        assert m.comorbidities == "Type 2 diabetes"

    def test_all_opd_fields_present(self):
        from model.department_fields import DEPARTMENT_FIELDS
        opd_keys = {f["key"] for f in DEPARTMENT_FIELDS["opd"]}
        schema_fields = set(OPDExtractionOutput.model_fields.keys())
        missing = opd_keys - schema_fields
        assert not missing, f"Missing OPD fields in schema: {missing}"

    def test_coercion_list_to_str(self):
        m = OPDExtractionOutput.model_validate({"allergy_status": ["NKDA", "penicillin"]})
        assert m.allergy_status == "NKDA, penicillin"

    def test_none_values_preserved(self):
        m = OPDExtractionOutput.model_validate({"chief_complaint": None})
        assert m.chief_complaint is None

    def test_extra_fields_ignored(self):
        m = OPDExtractionOutput.model_validate({
            "chief_complaint": "Fever",
            "triage_priority": "Category 1",  # ER-only field
        })
        assert m.chief_complaint == "Fever"
        assert not hasattr(m, "triage_priority")


# ══════════════════════════════════════════════════════════════════════════════
# §9  DentalExtractionOutput
# ══════════════════════════════════════════════════════════════════════════════

class TestDentalExtractionOutput:
    """DentalExtractionOutput — Dental department extraction output schema."""

    def test_valid_baseline(self):
        data = {
            "chief_complaint": "Severe toothache",
            "tooth_number": "FDI 36",
            "subspecialty_focus": "Endodontics",
            "procedure_executed": "Root canal therapy step 1",
            "consent_obtained": "Yes",
            "treatment_plan_status": "In-Progress",
        }
        m = DentalExtractionOutput.model_validate(data)
        assert m.tooth_number == "FDI 36"
        assert m.subspecialty_focus == "Endodontics"

    def test_all_dental_fields_present(self):
        from model.department_fields import DEPARTMENT_FIELDS
        dental_keys = {f["key"] for f in DEPARTMENT_FIELDS["dental"]}
        schema_fields = set(DentalExtractionOutput.model_fields.keys())
        missing = dental_keys - schema_fields
        assert not missing, f"Missing Dental fields in schema: {missing}"

    def test_list_coerced_to_str(self):
        m = DentalExtractionOutput.model_validate({
            "materials_inventory": ["Sodium hypochlorite", "Cavit"]
        })
        assert m.materials_inventory == "Sodium hypochlorite, Cavit"

    def test_all_fields_default_none(self):
        m = DentalExtractionOutput()
        assert m.chief_complaint is None
        assert m.followup_timeline is None


# ══════════════════════════════════════════════════════════════════════════════
# §10  DiagnosisOutput
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagnosisOutput:
    """DiagnosisOutput — ICD-10 recommendation output schema."""

    def test_valid_opd_baseline(self):
        data = {
            "icd10_codes": ["J18.9 - Pneumonia", "J20.9 - Bronchitis"],
            "reasoning": "Fever and crackles favour pneumonia.",
            "history_considered": "No",
            "history_notes": "",
        }
        m = DiagnosisOutput.model_validate(data)
        assert len(m.icd10_codes) == 2
        assert m.reasoning == "Fever and crackles favour pneumonia."
        assert m.history_considered == "No"
        assert m.differential_diagnoses is None

    def test_valid_er_baseline(self):
        data = {
            "icd10_codes": ["K35.80 - Acute appendicitis"],
            "reasoning": "RLQ tenderness with elevated WBC.",
            "history_considered": "No",
            "differential_diagnoses": ["K37 - Appendicitis unspecified"],
            "triage_priority": "Category 2 - Emergency",
        }
        m = DiagnosisOutput.model_validate(data)
        assert m.triage_priority == "Category 2 - Emergency"
        assert len(m.differential_diagnoses) == 1

    def test_icd10_codes_coercion_from_str(self):
        m = DiagnosisOutput.model_validate({"icd10_codes": "J18.9"})
        assert m.icd10_codes == ["J18.9"]

    def test_icd10_codes_coercion_from_dict(self):
        m = DiagnosisOutput.model_validate({
            "icd10_codes": {"1": "J18.9", "2": "J20.9"}
        })
        assert set(m.icd10_codes) == {"J18.9", "J20.9"}

    def test_icd10_codes_none_becomes_empty_list(self):
        m = DiagnosisOutput.model_validate({"icd10_codes": None})
        assert m.icd10_codes == []

    def test_reasoning_coercion(self):
        m = DiagnosisOutput.model_validate({"reasoning": None})
        assert m.reasoning == ""

    def test_final_diagnosis_optional(self):
        m = DiagnosisOutput.model_validate({"icd10_codes": []})
        assert m.final_diagnosis is None

    def test_triage_priority_coercion_blank_to_none(self):
        m = DiagnosisOutput.model_validate({"triage_priority": "  "})
        # blank string → empty string → None via coerce_opt_str
        assert m.triage_priority is None or m.triage_priority == ""

    def test_history_considered_default(self):
        m = DiagnosisOutput()
        assert m.history_considered == "No"
        assert m.history_notes == ""


# ══════════════════════════════════════════════════════════════════════════════
# §11  UsageMeta
# ══════════════════════════════════════════════════════════════════════════════

class TestUsageMeta:
    """UsageMeta — LLM call metadata."""

    def test_all_none_default(self):
        m = UsageMeta()
        assert m.model_id is None
        assert m.latency_sec is None
        assert m.cost_usd is None

    def test_valid_full_metadata(self):
        data = {
            "model_id": "DeepSeek-V3",
            "foundry_deployment": "DeepSeek-V3.2",
            "latency_sec": 1.234,
            "prompt_tokens": 512,
            "completion_tokens": 256,
            "reasoning_tokens": 64,
            "cost_usd": 0.001,
            "reasoning_enabled": True,
        }
        m = UsageMeta.model_validate(data)
        assert m.model_id == "DeepSeek-V3"
        assert m.latency_sec == 1.234
        assert m.reasoning_enabled is True

    def test_extra_keys_ignored(self):
        m = UsageMeta.model_validate({"model_id": "X", "unknown": "ignored"})
        assert m.model_id == "X"

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            UsageMeta.model_validate({"latency_sec": -0.1})

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            UsageMeta.model_validate({"prompt_tokens": -1})


# ══════════════════════════════════════════════════════════════════════════════
# §12  QuestionItem
# ══════════════════════════════════════════════════════════════════════════════

class TestQuestionItem:
    """QuestionItem — single doctor question."""

    def test_valid_baseline(self):
        m = QuestionItem(question="What medications is the patient taking?", category="History")
        assert m.needs_asking is True
        assert m.answer is None

    def test_answered_question(self):
        m = QuestionItem(
            question="What is the onset?",
            answer="3 days ago",
            needs_asking=False,
            category="History",
        )
        assert m.answer == "3 days ago"
        assert m.needs_asking is False

    def test_missing_required_question_rejected(self):
        with pytest.raises(ValidationError):
            QuestionItem(category="History")  # question is required

    def test_missing_required_category_rejected(self):
        with pytest.raises(ValidationError):
            QuestionItem(question="When did it start?")  # category is required


# ══════════════════════════════════════════════════════════════════════════════
# §13  PipelineSessionOutput
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineSessionOutput:
    """PipelineSessionOutput — full session envelope."""

    _REQUIRED = {"department": "opd", "language": "en", "mode": "doctor"}

    def test_valid_minimal(self):
        m = PipelineSessionOutput(**self._REQUIRED)
        assert m.status == "ok"
        assert m.specialty is None
        assert m.diagnosis is None

    def test_valid_full(self):
        m = PipelineSessionOutput(
            **self._REQUIRED,
            session_id="sess-001",
            status="ok",
            specialty="Cardiology",
            transcription_sec=15.0,
            refinement_sec=2.3,
            translation_sec=None,
            extraction_sec=4.1,
            question_generation_sec=0.05,
            diagnosis_sec=6.2,
            total_sec=27.65,
            transcript="Patient with chest pain",
            refined_text="Patient presents with chest pain",
            patient_story="Patient presents with chest pain",
            extracted_features={"chief_complaint": "Chest pain"},
            questions=[QuestionItem(question="Any prior episodes?", category="History")],
            diagnosis=DiagnosisOutput(
                icd10_codes=["I20.9"],
                reasoning="Typical angina pattern.",
                history_considered="No",
            ),
            extraction_usage=UsageMeta(latency_sec=4.1, prompt_tokens=300),
            diagnosis_usage=UsageMeta(latency_sec=6.2, prompt_tokens=400),
        )
        assert m.session_id == "sess-001"
        assert m.extraction_sec == 4.1
        assert len(m.questions) == 1
        assert m.diagnosis.icd10_codes == ["I20.9"]

    def test_missing_required_department_rejected(self):
        with pytest.raises(ValidationError):
            PipelineSessionOutput(language="en", mode="doctor")

    def test_missing_required_language_rejected(self):
        with pytest.raises(ValidationError):
            PipelineSessionOutput(department="opd", mode="doctor")

    def test_negative_timing_rejected(self):
        with pytest.raises(ValidationError):
            PipelineSessionOutput(**self._REQUIRED, total_sec=-1.0)

    def test_extra_keys_ignored(self):
        m = PipelineSessionOutput(**self._REQUIRED, unknown_field="ignored")  # type: ignore[call-arg]
        # extra="ignore" means no error, field just not set
        assert not hasattr(m, "unknown_field")


# ══════════════════════════════════════════════════════════════════════════════
# §14  EXTRACTION_SCHEMAS registry
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractionSchemasRegistry:
    """EXTRACTION_SCHEMAS and get_extraction_output_schema."""

    def test_er_maps_to_er_schema(self):
        assert EXTRACTION_SCHEMAS["er"] is ERExtractionOutput

    def test_opd_maps_to_opd_schema(self):
        assert EXTRACTION_SCHEMAS["opd"] is OPDExtractionOutput

    def test_dental_maps_to_dental_schema(self):
        assert EXTRACTION_SCHEMAS["dental"] is DentalExtractionOutput

    def test_helper_er(self):
        assert get_extraction_output_schema("er") is ERExtractionOutput

    def test_helper_opd(self):
        assert get_extraction_output_schema("opd") is OPDExtractionOutput

    def test_helper_dental(self):
        assert get_extraction_output_schema("dental") is DentalExtractionOutput

    def test_helper_unknown_falls_back_to_opd(self):
        assert get_extraction_output_schema("radiology") is OPDExtractionOutput

    def test_helper_empty_string_falls_back_to_opd(self):
        assert get_extraction_output_schema("") is OPDExtractionOutput

    def test_helper_case_insensitive(self):
        assert get_extraction_output_schema("ER") is ERExtractionOutput
        assert get_extraction_output_schema("OPD") is OPDExtractionOutput


# ══════════════════════════════════════════════════════════════════════════════
# §15  Field-key parity with department_fields.py (guard against drift)
# ══════════════════════════════════════════════════════════════════════════════

class TestDepartmentFieldParity:
    """Assert schemas stay in sync with DEPARTMENT_FIELDS — the single source
    of truth for field keys used by QuestionGenerator, the UI layout, and
    the Word templates."""

    def _get_df_keys(self, dept: str) -> set:
        from model.department_fields import DEPARTMENT_FIELDS
        return {f["key"] for f in DEPARTMENT_FIELDS[dept]}

    def test_er_keys_complete(self):
        missing = self._get_df_keys("er") - set(ERExtractionOutput.model_fields)
        assert not missing, f"ER schema missing keys from department_fields.py: {missing}"

    def test_opd_keys_complete(self):
        missing = self._get_df_keys("opd") - set(OPDExtractionOutput.model_fields)
        assert not missing, f"OPD schema missing keys from department_fields.py: {missing}"

    def test_dental_keys_complete(self):
        missing = self._get_df_keys("dental") - set(DentalExtractionOutput.model_fields)
        assert not missing, f"Dental schema missing keys from department_fields.py: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# §16  Performance — sub-millisecond validation budget
# ══════════════════════════════════════════════════════════════════════════════

class TestValidationPerformance:
    """Each hot-path schema validates in < 1ms on average."""

    _ER_DATA = {
        "chief_complaint": "Chest pain",
        "onset": "1 hour ago",
        "airway": "Patent",
        "breathing": "Clear",
        "circulation": "BP 130/80",
        "disability": "GCS 15",
        "plan_of_care": "ECG, Troponin",
    }
    _ANALYZE_DATA = {
        "text": "Patient with chest pain since 2 hours.",
        "language": "en",
        "mode": "doctor",
        "department": "er",
        "use_history": False,
    }

    def _measure(self, fn, iterations: int = 1000) -> float:
        # Warm-up
        for _ in range(50):
            fn()
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        return (time.perf_counter() - t0) * 1000 / iterations  # ms/call

    def test_er_extraction_output_sub_ms(self):
        avg = self._measure(lambda: ERExtractionOutput.model_validate(self._ER_DATA))
        assert avg < 1.0, f"ERExtractionOutput avg={avg:.4f}ms exceeded 1ms budget"

    def test_opd_extraction_output_sub_ms(self):
        data = {"chief_complaint": "Cough", "plan_of_care": "Rest"}
        avg = self._measure(lambda: OPDExtractionOutput.model_validate(data))
        assert avg < 1.0, f"OPDExtractionOutput avg={avg:.4f}ms exceeded 1ms budget"

    def test_analyze_input_sub_ms(self):
        avg = self._measure(lambda: AnalyzeInput.model_validate(self._ANALYZE_DATA))
        assert avg < 1.0, f"AnalyzeInput avg={avg:.4f}ms exceeded 1ms budget"

    def test_diagnosis_output_sub_ms(self):
        data = {"icd10_codes": ["J18.9"], "reasoning": "Pneumonia."}
        avg = self._measure(lambda: DiagnosisOutput.model_validate(data))
        assert avg < 1.0, f"DiagnosisOutput avg={avg:.4f}ms exceeded 1ms budget"


# ══════════════════════════════════════════════════════════════════════════════
# §17  Enum backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestEnumBackwardCompatibility:
    """Enum string values must match what the existing app.py / frontend sends."""

    def test_language_values(self):
        assert Language("ar") is Language.arabic
        assert Language("en") is Language.english

    def test_mode_values(self):
        assert Mode("doctor") is Mode.doctor
        assert Mode("conversation") is Mode.conversation

    def test_department_values(self):
        assert Department("er") is Department.er
        assert Department("opd") is Department.opd
        assert Department("dental") is Department.dental

    def test_business_unit_values(self):
        assert BusinessUnit("AHJ") is BusinessUnit.AHJ
        assert BusinessUnit("AMH") is BusinessUnit.AMH
        assert BusinessUnit("ANC") is BusinessUnit.ANC

    def test_language_in_analyze_input(self):
        """Ensure AnalyzeInput accepts raw string values (as the form sends)."""
        m = AnalyzeInput(text="hello patient", language="ar")
        assert m.language == Language.arabic

    def test_department_in_analyze_input(self):
        m = AnalyzeInput(text="hello patient", department="dental")
        assert m.department == Department.dental
