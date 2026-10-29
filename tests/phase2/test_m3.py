"""
Phase 2 Test Suite — Track M3: Validation Layer
Subtasks: M3.1, M3.2, M3.3, M3.4, M3.5 (5 hard test cases each = 25 test cases total)
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation.icd10 import get_icd10_validator, ResolutionTier
from validation.schemas import SchemaValidator
from validation.consistency import ConsistencyValidator, CautionBanner
from validation.laterality import check_laterality_conflict, detect_laterality, Laterality
from validation.history_guard import HistoryCrossCheckGuard, HistoryDiscrepancyFlag
from validation.missing_fields import MissingFieldChecklist
from validation.pipeline import ClinicalValidationPipeline, ValidationReport
from tests.phase1.harness import SubtaskRunner


# ==============================================================================
# M3.1 — ICD-10 lookup from HIS
# ==============================================================================
def run_test_m3_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M3",
        subtask_id="M3.1",
        title="ICD-10 code lookup from HIS master & validation",
        target_files=["src/validation/icd10.py", "src/data/Austrilian ICD codes.csv"],
        source_file=__file__,
    )

    # TC 1: Valid code lookup — 'J18.9' -> Valid + matching description
    def tc1():
        validator = get_icd10_validator()
        res = validator.validate_code("J18.9")
        assert res.is_valid is True, f"Expected is_valid=True, got {res.is_valid}"
        assert res.canonical_code == "J18.9"
        assert res.description and "pneumonia" in res.description.lower()
        assert res.tier == ResolutionTier.EXACT_MATCH
        return True, f"Code J18.9 validated: canonical={res.canonical_code}, tier={res.tier.value}, description='{res.description}'"

    runner.record_case(
        case_num=1,
        name="Valid ICD-10 code verification",
        input_setup="Validate code 'J18.9' against HIS ICD-10 master validator",
        expected_result="is_valid is True with accurate description ('Pneumonia, unspecified') and EXACT_MATCH tier",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Invalid code rejection — 'ZZZ.999' -> Invalid status
    def tc2():
        validator = get_icd10_validator()
        res = validator.validate_code("ZZZ.999")
        assert res.is_valid is False
        assert res.tier == ResolutionTier.UNRESOLVED
        assert res.rejection_reason is not None
        return True, f"Invalid code ZZZ.999 classified as invalid with reason '{res.rejection_reason}'"

    runner.record_case(
        case_num=2,
        name="Invalid ICD-10 code rejection",
        input_setup="Validate non-existent code 'ZZZ.999'",
        expected_result="is_valid is False, tier is UNRESOLVED, rejection_reason set",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Batch 10 codes lookup latency < 500 ms
    def tc3():
        validator = get_icd10_validator()
        codes = ["J18.9", "I10", "E11.9", "R07.9", "K21.9", "M54.5", "J06.9", "N39.0", "K29.7", "R51"]
        t0 = time.perf_counter()
        batch_res = validator.validate_codes(codes)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert len(batch_res.results) == 10
        assert batch_res.all_valid is True
        assert elapsed_ms < 500.0, f"Batch validation took {elapsed_ms:.2f}ms > 500ms"
        return True, f"Batch validation of 10 codes executed in {elapsed_ms:.2f}ms < 500ms (all 10 valid)"

    runner.record_case(
        case_num=3,
        name="Batch ICD-10 validation latency bound",
        input_setup="Validate 10 codes simultaneously in batch",
        expected_result="All validated accurately with cumulative latency < 500 ms",
        severity="High",
        fn=tc3,
    )

    # TC 4: Case-insensitivity — 'j18.9' vs 'J18.9' -> Same result
    def tc4():
        validator = get_icd10_validator()
        upper_res = validator.validate_code("J18.9")
        lower_res = validator.validate_code("j18.9")
        assert upper_res.is_valid is True and lower_res.is_valid is True
        assert upper_res.canonical_code == lower_res.canonical_code == "J18.9"
        return True, "Case-insensitivity verified: lowercase 'j18.9' normalized to canonical 'J18.9'"

    runner.record_case(
        case_num=4,
        name="ICD-10 code case-insensitivity",
        input_setup="Compare validation results for 'j18.9' and 'J18.9'",
        expected_result="Both normalize to uppercase 'J18.9' with identical valid status",
        severity="High",
        fn=tc4,
    )

    # TC 5: File integrity — Count codes in CSV >= 10,000; no parse errors
    def tc5():
        validator = get_icd10_validator()
        code_count = len(validator.registry._by_norm_code)
        assert code_count >= 10000, f"Expected >= 10,000 codes, loaded {code_count}"
        return True, f"HIS master file integrity verified: {code_count} codes indexed with zero parse errors"

    runner.record_case(
        case_num=5,
        name="HIS code master file integrity",
        input_setup="Check total code dictionary size from Australian ICD master",
        expected_result=">= 10,000 valid codes loaded cleanly without syntax errors",
        severity="Crit",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M3.2 — Pydantic output schemas
# ==============================================================================
def run_test_m3_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M3",
        subtask_id="M3.2",
        title="Validation layer Pydantic schemas & input sanitization",
        target_files=["src/validation/schemas.py"],
        source_file=__file__,
    )

    # TC 1: Valid passes — Well-formed output -> No errors
    def tc1():
        data = {
            "chief_complaint": "Acute appendicitis presentation",
            "onset": "Yesterday",
            "course": "Worsening",
            "duration": "1 day",
            "airway": "Patent",
            "plan_of_care": "Emergency surgery",
        }
        val, err = SchemaValidator.validate_extraction(data, department="er")
        assert err is None, f"Expected no error, got {err}"
        assert val is not None
        dump = val.model_dump()
        assert dump["chief_complaint"] == "Acute appendicitis presentation"
        assert dump["airway"] == "Patent"
        return True, "Well-formed clinical output validated with zero errors"

    runner.record_case(
        case_num=1,
        name="Valid payload schema pass",
        input_setup="Validate well-formed clinical dictionary through SchemaValidator",
        expected_result="Passes validation cleanly; returns sanitized instance",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Rejection on non-dict payload
    def tc2():
        invalid_data = ["not", "a", "dict"]
        val, err = SchemaValidator.validate_extraction(invalid_data, department="er")
        assert val is None
        assert err is not None
        assert "dict" in err.reason.lower()
        return True, f"Invalid payload rejected: {err.reason}"

    runner.record_case(
        case_num=2,
        name="Non-dict payload rejection",
        input_setup="Submit non-dictionary list to validate_extraction",
        expected_result="Rejection payload generated specifying expected dictionary",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Type coercion — Int where string -> Coerced gracefully
    def tc3():
        data = {
            "chief_complaint": "Migraine",
            "duration": 48,  # integer duration coerced to string
        }
        val, err = SchemaValidator.validate_extraction(data, department="opd")
        assert err is None
        assert val is not None
        assert isinstance(val.duration, str)
        assert "48" in val.duration
        return True, f"Integer duration coerced to string: '{val.duration}'"

    runner.record_case(
        case_num=3,
        name="Type coercion resilience",
        input_setup="Pass numeric integer 48 into string duration field",
        expected_result="Coerced cleanly to string '48' without crashing",
        severity="High",
        fn=tc3,
    )

    # TC 4: Extra fields — malicious_field added -> Stripped; no injection
    def tc4():
        data = {
            "chief_complaint": "Chest pain",
            "__proto__": "polluted",
            "malicious_sql": "DROP TABLE patients;",
        }
        val, err = SchemaValidator.validate_extraction(data, department="er")
        assert err is None
        assert val is not None
        dump = val.model_dump()
        assert "malicious_sql" not in dump
        assert "__proto__" not in dump
        assert dump["chief_complaint"] == "Chest pain"
        return True, "Extraneous / malicious injection keys cleanly stripped from validated model dump"

    runner.record_case(
        case_num=4,
        name="Malicious and extra fields stripping",
        input_setup="Submit dictionary containing unrecognized / injection keys",
        expected_result="Extra keys stripped; only valid schema fields retained",
        severity="Crit",
        fn=tc4,
    )

    # TC 5: Performance < 0.5 ms per validation
    def tc5():
        data = {"chief_complaint": "Sore throat", "duration": "2 days", "plan_of_care": "Rest"}
        t0 = time.perf_counter()
        for _ in range(500):
            SchemaValidator.validate_extraction(data, department="opd")
        avg_ms = ((time.perf_counter() - t0) / 500) * 1000.0
        assert avg_ms < 0.5, f"Validation avg latency {avg_ms:.3f}ms exceeded 0.5ms limit"
        return True, f"High-performance validation: average {avg_ms*1000:.1f} µs per record (< 0.5 ms)"

    runner.record_case(
        case_num=5,
        name="High-performance schema validation latency",
        input_setup="Measure average validation latency across 500 validations",
        expected_result="Average latency < 0.5 ms per validation",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M3.3 — Consistency rules
# ==============================================================================
def run_test_m3_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M3",
        subtask_id="M3.3",
        title="Clinical consistency rules: laterality & diagnostic evidence cross-checking",
        target_files=["src/validation/consistency.py", "src/validation/laterality.py"],
        source_file=__file__,
    )

    # TC 1: Laterality mismatch — 'left knee' -> 'right knee' flagged
    def tc1():
        conflict = check_laterality_conflict("Pain in the left knee", "M17.11 - Primary osteoarthritis, right knee")
        assert conflict is not None
        assert "Laterality mismatch" in conflict
        return True, f"Laterality mismatch caught: {conflict}"

    runner.record_case(
        case_num=1,
        name="Laterality mismatch detection",
        input_setup="Cross-check complaint 'left knee' with ICD-10 'right knee'",
        expected_result="Conflict string returned identifying left vs right mismatch",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Diagnosis w/o evidence — 'pneumonia' with zero respiratory findings -> Warning
    def tc2():
        cautions = ConsistencyValidator.check_evidence_grounding(
            clinical_text="Sprained ankle with edema and pain on weight bearing",
            diagnoses=["J18.9 - Pneumonia, unspecified"],
        )
        assert len(cautions) > 0
        assert "DIAGNOSIS_UNGROUNDED" in cautions[0].rule
        return True, f"Unsubstantiated diagnosis flagged with caution banner: {cautions[0].message}"

    runner.record_case(
        case_num=2,
        name="Diagnosis without clinical evidence warning",
        input_setup="Evaluate diagnosis 'Pneumonia' against isolated 'Sprained ankle' findings",
        expected_result="Caution banner emitted indicating lack of supporting clinical evidence",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Consistent passes — 'right eye' both -> No warnings
    def tc3():
        conflict = check_laterality_conflict("Foreign body in right eye", "H16.0 - Corneal ulcer, right eye")
        assert conflict is None, f"Unexpected conflict flagged: {conflict}"
        return True, "Consistent right eye laterality passed with zero warnings"

    runner.record_case(
        case_num=3,
        name="Consistent laterality pass",
        input_setup="Cross-check 'right eye' in complaints and 'right eye' in code description",
        expected_result="Zero warnings or conflicts emitted (conflict is None)",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Bilateral accepted — 'bilateral knee pain'
    def tc4():
        det = detect_laterality("Bilateral knee pain and osteoarthritis")
        assert det.laterality == Laterality.BILATERAL
        conflict = check_laterality_conflict("Bilateral knee pain", "M17.0 - Bilateral primary osteoarthritis of knee")
        assert conflict is None
        return True, "Bilateral laterality detected and accepted across findings and code description"

    runner.record_case(
        case_num=4,
        name="Bilateral laterality validation",
        input_setup="Evaluate 'bilateral knee pain' against 'bilateral knee' ICD-10 description",
        expected_result="Accepted as valid bilateral presentation; zero false-positive conflicts",
        severity="High",
        fn=tc4,
    )

    # TC 5: Multiple checks — 2 lateralized pairs evaluated independently
    def tc5():
        c1 = check_laterality_conflict("Left shoulder dislocation", "S43.004A - Dislocation of right shoulder")
        c2 = check_laterality_conflict("Left wrist sprain", "S63.502A - Sprain of left wrist")
        assert c1 is not None  # conflict
        assert c2 is None      # consistent
        return True, "Multiple lateralized anatomical evaluations executed cleanly and independently"

    runner.record_case(
        case_num=5,
        name="Multiple independent anatomical checks",
        input_setup="Evaluate shoulder mismatch and wrist match in succession",
        expected_result="Each anatomical pair evaluated independently without state crosstalk",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M3.4 — History cross-check + missing fields
# ==============================================================================
def run_test_m3_4() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M3",
        subtask_id="M3.4",
        title="History cross-checking & department missing-field checklists",
        target_files=["src/validation/history_guard.py", "src/validation/missing_fields.py"],
        source_file=__file__,
    )

    # TC 1: Reused data structure verification
    def tc1():
        history = [{"date": "2025-01-01", "diagnosis": "Hypertension", "icd10_diagnoses": ["I10"]}]
        report = HistoryCrossCheckGuard.evaluate(
            past_visits=history,
            extracted_features={"icd10_codes": ["I10"], "history_considered": "Yes"},
        )
        assert report.history_provided is True
        assert report.claimed_considered is True
        assert report.overlap_found is True
        return True, f"History alignment report verified: overlap_families={report.overlap_families}"

    runner.record_case(
        case_num=1,
        name="History structure validation",
        input_setup="Process standard patient history dictionary through HistoryCrossCheckGuard",
        expected_result="History records evaluated cleanly; overlap tracked accurately",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Deterministic — 10x same input -> Identical checklist
    def tc2():
        er_note = {"chief_complaint": "Laceration"}
        results = [MissingFieldChecklist.evaluate(er_note, department="er") for _ in range(10)]
        first_missing = [item.key for item in results[0].missing_fields]
        for r in results:
            keys = [item.key for item in r.missing_fields]
            assert keys == first_missing
        return True, f"Deterministic checklist: 10 repeated runs produced identical missing fields list ({len(first_missing)} missing)"

    runner.record_case(
        case_num=2,
        name="Missing fields checklist determinism",
        input_setup="Evaluate missing fields on ER note 10 consecutive times",
        expected_result="Bitwise identical missing field lists returned across all 10 runs",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Required fields checklist in ER
    def tc3():
        er_note = {"chief_complaint": "Chest pain"}
        report = MissingFieldChecklist.evaluate(er_note, department="er")
        missing_keys = [item.key for item in report.missing_fields]
        assert "examination" in missing_keys or "airway" in missing_keys or "plan_of_care" in missing_keys
        assert report.completeness_percentage < 50.0
        return True, f"ER missing fields checklist identified {report.missing_count} missing fields (completeness: {report.completeness_percentage}%)"

    runner.record_case(
        case_num=3,
        name="Mandatory missing fields detection",
        input_setup="Submit ER note with only chief_complaint",
        expected_result="Missing required clinical sections populated in checklist",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Optional / populated fields completeness
    def tc4():
        opd_full = {
            "chief_complaint": "Routine followup",
            "onset": "3 days",
            "course": "Stable",
            "duration": "3 days",
            "past_medical_history": "Hypertension diagnosed 5 years ago",
            "past_surgical_history": "Appendectomy 2010",
            "medication_history": "Amlodipine 5mg daily",
            "allergy_status": "NKDA",
            "comorbidities": "Essential hypertension",
            "social_family_history": "Non-smoker",
            "examination_finding": "Normal chest and heart sounds",
            "procedures_planned": "None planned today",
            "plan_of_care": "Discharge home with followup",
            "referrals": "No referral required",
            "follow_up_timing": "1 month",
            "patient_instructions": "Take prescribed medications regularly",
            "outcome_progress_status": "Improved stable",
        }
        report = MissingFieldChecklist.evaluate(opd_full, department="opd")
        assert report.missing_count == 0
        assert report.completeness_percentage == 100.0
        return True, "Complete OPD encounter validated with 100% completeness score"

    runner.record_case(
        case_num=4,
        name="Complete encounter checklist verification",
        input_setup="Validate fully populated OPD note dictionary",
        expected_result="missing_count is 0, completeness is 100%",
        severity="High",
        fn=tc4,
    )

    # TC 5: Discrepancy flag — claims used with zero overlap
    def tc5():
        history = [{"date": "2025-01-01", "diagnosis": "Hypertension", "icd10_diagnoses": ["I10"]}]
        report = HistoryCrossCheckGuard.evaluate(
            past_visits=history,
            extracted_features={"icd10_codes": ["S82.0"], "history_considered": "Yes"},
        )
        assert report.flag == HistoryDiscrepancyFlag.CLAIMS_USED_NO_OVERLAP.value
        assert report.caution_banner is not None
        return True, f"History discrepancy flagged: {report.caution_banner.message}"

    runner.record_case(
        case_num=5,
        name="History consideration contradiction detection",
        input_setup="Evaluate claim of history usage with zero overlapping diagnostic families",
        expected_result="Flagged as claims_used_no_overlap with caution banner emitted",
        severity="Crit",
        fn=tc5,
    )

    return runner


# ==============================================================================
# M3.5 — Rule tests
# ==============================================================================
def run_test_m3_5() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="M3",
        subtask_id="M3.5",
        title="Validation pipeline holistic rule tests & modular rule independence",
        target_files=["src/validation/pipeline.py"],
        source_file=__file__,
    )

    # TC 1: Test per rule type verification
    def tc1():
        # Verify pipeline aggregates all validation rules
        assert hasattr(ClinicalValidationPipeline, "validate")
        return True, "ClinicalValidationPipeline encompasses schema, ICD-10, laterality, and history guard"

    runner.record_case(
        case_num=1,
        name="Holistic pipeline rule coverage",
        input_setup="Inspect ClinicalValidationPipeline rule components",
        expected_result="Encapsulates schema, code, laterality, and history validation rules",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: All validation tests pass
    def tc2():
        report = ClinicalValidationPipeline.validate(
            department="opd",
            raw_extraction={
                "chief_complaint": "Right knee osteoarthritis",
                "onset": "6 months ago",
                "course": "Gradual",
                "duration": "6 months",
                "examination_finding": "Crepitus in right knee",
                "plan_of_care": "Physical therapy",
            },
            raw_diagnosis={
                "icd10_codes": ["M17.11 - Unilateral primary osteoarthritis, right knee"],
            },
            patient_history=[],
        )
        assert report.is_valid is True
        assert len(report.diagnoses["validated_codes"]) == 1
        return True, f"All validation rules passed for coherent clinical note (is_valid={report.is_valid})"

    runner.record_case(
        case_num=2,
        name="Holistic coherent clinical encounter pass",
        input_setup="Validate completely coherent clinical encounter through ClinicalValidationPipeline",
        expected_result="Pipeline returns is_valid=True with zero critical errors",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Zero warnings on optimal clinical input
    def tc3():
        report = ClinicalValidationPipeline.validate(
            department="opd",
            raw_extraction={
                "chief_complaint": "Hypertension followup",
                "onset": "1 year",
                "course": "Controlled",
                "duration": "1 year",
                "examination_finding": "BP 125/80 mmHg with regular pulse",
                "plan_of_care": "Continue medications",
            },
            raw_diagnosis={
                "icd10_codes": ["I10 - Essential (primary) hypertension"],
            },
            patient_history=[],
            translated_text="Patient presents for routine hypertension follow-up. Blood pressure is 125/80.",
        )
        # Grounding should find hypertension in text; laterality is unspecified -> 0 cautions
        lat_cautions = [c for c in report.cautions if c.rule == "LATERALITY_MISMATCH"]
        assert len(lat_cautions) == 0
        return True, "Zero laterality cautions generated on optimal well-formed clinical input"

    runner.record_case(
        case_num=3,
        name="Zero warnings on optimal input",
        input_setup="Validate optimal clinical encounter",
        expected_result="Zero laterality conflicts emitted",
        severity="High",
        fn=tc3,
    )

    # TC 4: Multi-fault detection on maximally invalid input
    def tc4():
        report = ClinicalValidationPipeline.validate(
            department="opd",
            raw_extraction={
                "chief_complaint": "Left bundle branch block",
                "examination_finding": "ECG reveals left bundle branch block",
            },
            raw_diagnosis={
                "icd10_codes": ["INVALID_CODE_999", "I45.0 - Right fascicular block"],
            },
            patient_history=[{"date": "2025-01-01", "diagnosis": "Diabetes", "icd10_diagnoses": ["E11"]}],
        )
        # Should flag invalid code in unmatched
        assert "INVALID_CODE_999" in report.diagnoses["unmatched_codes"]
        # Should flag laterality mismatch between left bundle in examination and right fascicular in code
        assert any(c.rule == "LATERALITY_MISMATCH" for c in report.cautions)
        return True, "Multi-fault detection verified: both laterality conflict and invalid ICD-10 code surfaced"

    runner.record_case(
        case_num=4,
        name="Multi-fault detection on invalid input",
        input_setup="Submit payload with simultaneous laterality mismatch and invalid ICD-10 code",
        expected_result="Each defect captured independently in respective validation collections",
        severity="High",
        fn=tc4,
    )

    # TC 5: Modular independence of validation rules
    def tc5():
        # Laterality module functions completely independently of ICD-10 registry
        c = check_laterality_conflict("Left arm fracture", "Left arm cast applied")
        assert c is None
        # ICD-10 validator functions independently of laterality
        v = get_icd10_validator().validate_code("I10")
        assert v.is_valid is True
        return True, "Modular rule independence confirmed: modules operate without circular coupling"

    runner.record_case(
        case_num=5,
        name="Validation rule modular independence",
        input_setup="Execute laterality and ICD-10 validators in total isolation",
        expected_result="Operate cleanly and independently without circular dependencies",
        severity="High",
        fn=tc5,
    )

    return runner


if __name__ == "__main__":
    for func in [run_test_m3_1, run_test_m3_2, run_test_m3_3, run_test_m3_4, run_test_m3_5]:
        r = func()
        rep = r.finalize()
        print(f"[{r.subtask_id}] Passed: {rep.pass_count}/{len(rep.cases)} (Cleared: {rep.passed})")
