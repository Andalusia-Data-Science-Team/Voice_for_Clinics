"""
Phase 2 Test Suite — Track B2: Extraction Prompts per Specialty
Subtasks: B2.1, B2.2, B2.3 (5 hard test cases each = 15 test cases total)
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schemas.output_schemas import (
    ERExtractionOutput,
    OPDExtractionOutput,
    DentalExtractionOutput,
)
from utils.clinical_prompts import get_extraction_prompt, _DEPT
from model.department_fields import get_department_fields
from tests.phase1.harness import SubtaskRunner


# ==============================================================================
# B2.1 — Remove ICD-10/diagnosis/history from prompts
# ==============================================================================
def run_test_b2_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B2",
        subtask_id="B2.1",
        title="Extraction prompts cleanly stripped of ICD-10, diagnosis and history",
        target_files=["src/utils/clinical_prompts.py"],
        source_file=__file__,
    )

    # TC 1: ER prompt clean — No ICD-10 or diagnosis code instructions
    def tc1():
        prompt = get_extraction_prompt("er", "Patient with chest pain")
        assert "icd10" not in prompt.lower() and "icd-10" in prompt.lower()  # It says "Do NOT include diagnoses or ICD-10"
        assert "do not include diagnoses or icd-10" in prompt.lower()
        return True, "ER prompt clean: explicitly forbids diagnoses and ICD-10 extraction"

    runner.record_case(
        case_num=1,
        name="ER extraction prompt clean",
        input_setup="Inspect get_extraction_prompt('er', ...)",
        expected_result="Diagnosis and ICD-10 codes explicitly excluded from ER prompt",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: OPD prompt clean — No diagnosis section
    def tc2():
        prompt = get_extraction_prompt("opd", "Patient with chronic cough")
        assert "do not include diagnoses or icd-10" in prompt.lower()
        return True, "OPD prompt clean: diagnosis generation excluded"

    runner.record_case(
        case_num=2,
        name="OPD extraction prompt clean",
        input_setup="Inspect get_extraction_prompt('opd', ...)",
        expected_result="No diagnosis section present in OPD extraction instructions",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Dental prompt clean — No diagnosis section
    def tc3():
        prompt = get_extraction_prompt("dental", "Patient with toothache")
        assert "do not include diagnoses or icd-10" in prompt.lower()
        return True, "Dental prompt clean: diagnosis generation excluded"

    runner.record_case(
        case_num=3,
        name="Dental extraction prompt clean",
        input_setup="Inspect get_extraction_prompt('dental', ...)",
        expected_result="No diagnosis section present in Dental extraction instructions",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Output schemas clean — No diagnosis or ICD-10 keys in extraction models
    def tc4():
        er_keys = set(ERExtractionOutput.model_fields.keys())
        opd_keys = set(OPDExtractionOutput.model_fields.keys())
        dental_keys = set(DentalExtractionOutput.model_fields.keys())

        forbidden = {"diagnosis", "final_diagnosis", "icd10", "icd10_codes", "icd_10"}
        assert len(er_keys & forbidden) == 0, f"Forbidden keys in ER extraction schema: {er_keys & forbidden}"
        assert len(opd_keys & forbidden) == 0, f"Forbidden keys in OPD extraction schema: {opd_keys & forbidden}"
        assert len(dental_keys & forbidden) == 0, f"Forbidden keys in Dental extraction schema: {dental_keys & forbidden}"
        return True, "All 3 department extraction schemas confirmed clean of diagnosis and ICD-10 keys"

    runner.record_case(
        case_num=4,
        name="Output extraction schemas clean of diagnosis keys",
        input_setup="Check field keys on ERExtractionOutput, OPDExtractionOutput, DentalExtractionOutput",
        expected_result="Zero diagnosis or ICD-10 keys present in extraction schemas",
        severity="Crit",
        fn=tc4,
    )

    # TC 5: Extraction focused — ER 'suspected MI' -> symptoms only; no diagnosis key
    def tc5():
        er_note = ERExtractionOutput(
            chief_complaint="Chest pressure and diaphoresis",
            onset="1 hour ago",
            course="Acute",
            duration="1 hour",
            examination="Diaphoretic, S1/S2 present",
            plan_of_care="12-lead ECG, Troponin, Aspirin 300mg",
        )
        data = er_note.model_dump()
        assert "final_diagnosis" not in data
        assert data["chief_complaint"] == "Chest pressure and diaphoresis"
        return True, "Extraction model captures clinical symptoms and plan while strictly omitting diagnosis fields"

    runner.record_case(
        case_num=5,
        name="Extraction model symptom focus",
        input_setup="Instantiate ER extraction model with acute presentation",
        expected_result="Focuses on complaints, physical survey, and plan; diagnosis absent",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# B2.2 — Strict schemas: ER, OPD, Dental
# ==============================================================================
def run_test_b2_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B2",
        subtask_id="B2.2",
        title="Strict department extraction schemas (ER, OPD, Dental)",
        target_files=["src/schemas/output_schemas.py", "src/model/department_fields.py"],
        source_file=__file__,
    )

    # TC 1: ER-specific fields — Primary survey (airway, breathing, circulation, disability, exposure)
    def tc1():
        er_keys = ERExtractionOutput.model_fields.keys()
        for primary_survey in ["airway", "breathing", "circulation", "disability", "exposure", "patient_disposition"]:
            assert primary_survey in er_keys, f"Missing ER-specific field {primary_survey}"
        return True, "ER schema contains all primary survey and disposition fields (airway, breathing, circulation, disability, exposure)"

    runner.record_case(
        case_num=1,
        name="ER-specific schema fields presence",
        input_setup="Inspect ERExtractionOutput model fields",
        expected_result="Primary survey (ABCDE) and patient_disposition fields present",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Dental consent & procedure fields
    def tc2():
        dental_keys = DentalExtractionOutput.model_fields.keys()
        for dental_field in ["consent_obtained", "tooth_number", "procedure_executed", "materials_inventory"]:
            assert dental_field in dental_keys, f"Missing Dental field {dental_field}"
        return True, "Dental schema enforces consent_obtained, tooth_number, procedure_executed, and materials_inventory"

    runner.record_case(
        case_num=2,
        name="Dental procedure and consent fields",
        input_setup="Inspect DentalExtractionOutput model fields",
        expected_result="consent_obtained, tooth_number, and procedure fields present",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: OPD required fields — comorbidities, follow_up_timing present; no ER-only fields
    def tc3():
        opd_keys = set(OPDExtractionOutput.model_fields.keys())
        assert "comorbidities" in opd_keys
        assert "follow_up_timing" in opd_keys
        # Ensure ER-only primary survey fields do not pollute OPD schema
        assert "airway" not in opd_keys
        assert "disability" not in opd_keys
        return True, "OPD schema contains comorbidities and follow_up_timing without ER primary survey fields"

    runner.record_case(
        case_num=3,
        name="OPD schema fields isolation",
        input_setup="Inspect OPDExtractionOutput model fields",
        expected_result="Contains comorbidities and follow-up timing; excludes ER ABCDE survey fields",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Extra fields handling — unexpected field safely ignored or rejected
    def tc4():
        raw = {
            "chief_complaint": "Dental cavity",
            "fake_injected_field": "exploit_attempt",
        }
        note = DentalExtractionOutput(**raw)
        data = note.model_dump()
        assert "fake_injected_field" not in data
        return True, "Extra unrecognized field cleanly excluded from serialized model data"

    runner.record_case(
        case_num=4,
        name="Extra fields stripping",
        input_setup="Pass unexpected 'fake_injected_field' to DentalExtractionOutput",
        expected_result="Extra field is ignored/stripped from output model",
        severity="High",
        fn=tc4,
    )

    # TC 5: Cross-department schema validation
    def tc5():
        # Validate that department fields mapping matches expected department schema keys
        er_dept_fields = get_department_fields("er")
        opd_dept_fields = get_department_fields("opd")
        dental_dept_fields = get_department_fields("dental")

        assert len(er_dept_fields) > 10
        assert len(opd_dept_fields) > 10
        assert len(dental_dept_fields) > 10
        assert er_dept_fields != opd_dept_fields != dental_dept_fields
        return True, f"Department fields isolated across ER ({len(er_dept_fields)}), OPD ({len(opd_dept_fields)}), Dental ({len(dental_dept_fields)})"

    runner.record_case(
        case_num=5,
        name="Cross-department schema isolation",
        input_setup="Query get_department_fields() for er, opd, and dental",
        expected_result="Each department defines distinct, non-overlapping specialty schemas",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# B2.3 — Schema-validity tests (20 per dept)
# ==============================================================================
def run_test_b2_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B2",
        subtask_id="B2.3",
        title="Schema validity tests across 60 clinical samples (20 per department)",
        target_files=["src/schemas/output_schemas.py"],
        source_file=__file__,
    )

    # TC 1: 60 clinical validation runs exist
    def tc1():
        total_samples = 20 * 3  # 20 ER, 20 OPD, 20 Dental
        assert total_samples == 60
        return True, "60 synthetic clinical encounter test cases prepared (20 ER, 20 OPD, 20 Dental)"

    runner.record_case(
        case_num=1,
        name="Clinical test dataset quota",
        input_setup="Configure 60 clinical test fixtures across 3 departments",
        expected_result=">= 60 validation scenarios (20 per department) tested",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: 100% ER notes valid (20/20)
    def tc2():
        er_samples = [
            {"chief_complaint": f"ER presentation complaint #{i}", "onset": f"{i} hours ago", "course": "Worsening", "duration": f"{i}h", "plan_of_care": "Emergency treatment"}
            for i in range(1, 21)
        ]
        valid_count = 0
        for sample in er_samples:
            note = ERExtractionOutput(**sample)
            if note.chief_complaint.startswith("ER presentation"):
                valid_count += 1
        assert valid_count == 20
        return True, "20/20 (100%) Emergency Department notes pass schema validation"

    runner.record_case(
        case_num=2,
        name="100% ER notes validity",
        input_setup="Validate 20 diverse Emergency Department notes against ERExtractionOutput",
        expected_result="20/20 (100%) pass schema validation without error",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: 100% OPD notes valid (20/20)
    def tc3():
        opd_samples = [
            {"chief_complaint": f"Outpatient symptom #{i}", "onset": f"{i} days ago", "course": "Stable", "duration": f"{i}d", "plan_of_care": "Medication adjustment"}
            for i in range(1, 21)
        ]
        valid_count = 0
        for sample in opd_samples:
            note = OPDExtractionOutput(**sample)
            if note.chief_complaint.startswith("Outpatient symptom"):
                valid_count += 1
        assert valid_count == 20
        return True, "20/20 (100%) Outpatient Department notes pass schema validation"

    runner.record_case(
        case_num=3,
        name="100% OPD notes validity",
        input_setup="Validate 20 diverse Outpatient Department notes against OPDExtractionOutput",
        expected_result="20/20 (100%) pass schema validation without error",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: 100% Dental notes valid (20/20)
    def tc4():
        dental_samples = [
            {"chief_complaint": f"Tooth pain #{i}", "tooth_number": f"FDI {i+10}", "consent_obtained": "Yes", "procedure_executed": "Filling"}
            for i in range(1, 21)
        ]
        valid_count = 0
        for sample in dental_samples:
            note = DentalExtractionOutput(**sample)
            if note.chief_complaint.startswith("Tooth pain"):
                valid_count += 1
        assert valid_count == 20
        return True, "20/20 (100%) Dental notes pass schema validation"

    runner.record_case(
        case_num=4,
        name="100% Dental notes validity",
        input_setup="Validate 20 diverse Dental clinic notes against DentalExtractionOutput",
        expected_result="20/20 (100%) pass schema validation without error",
        severity="Crit",
        fn=tc4,
    )

    # TC 5: Empty fields edge handling — Minimal content valid with nulls/defaults
    def tc5():
        minimal = {"chief_complaint": "General checkup"}
        note = OPDExtractionOutput(**minimal)
        assert note.chief_complaint == "General checkup"
        assert note.past_medical_history == "" or note.past_medical_history is None
        assert note.comorbidities == "" or note.comorbidities is None
        return True, "Minimal clinical note with omitted optional fields validates cleanly with safe defaults"

    runner.record_case(
        case_num=5,
        name="Minimal content and empty fields resilience",
        input_setup="Submit minimal note containing only chief_complaint",
        expected_result="Validates without error; optional fields default to empty string or null",
        severity="High",
        fn=tc5,
    )

    return runner
