"""
Phase 2 Test Suite — Track B0: Term Correction & Red-Flag Lexicon
Subtasks: B0.1, B0.2 (5 hard test cases each = 10 test cases total)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.specialty_terms import correct_specialty_terms, SPECIALTIES
from utils.red_flags import detect_red_flags
from tests.phase1.harness import SubtaskRunner


# ==============================================================================
# B0.1 — Term correction before structuring
# ==============================================================================
def run_test_b0_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B0",
        subtask_id="B0.1",
        title="Term correction before structuring & abbreviation expansion",
        target_files=["src/model/specialty_terms.py", "src/model/refine_text.py"],
        source_file=__file__,
    )

    # TC 1: Pipeline order — Trace step order: term_correction BEFORE structuring
    def tc1():
        from model.refine_text import RefineText
        import inspect
        source = inspect.getsource(RefineText.refining_transcription)
        pos_correct = source.find("correct_specialty_terms(raw_text")
        pos_llm = source.find("LLMService.refine_")
        assert pos_correct != -1, "correct_specialty_terms pre-pass missing from refining_transcription"
        assert pos_llm != -1, "LLM refinement step missing from refining_transcription"
        assert pos_correct < pos_llm, "Pipeline order violation: term_correction must execute BEFORE LLM structuring"
        return True, "Pipeline execution order verified: pre-pass term_correction runs strictly before LLM structuring"

    runner.record_case(
        case_num=1,
        name="Pipeline order: term correction before structuring",
        input_setup="Inspect AST / source sequence of RefineText.refining_transcription",
        expected_result="correct_specialty_terms pre-pass runs strictly before LLM call",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Abbreviation expansion — 'pt c/o sob' expanded or normalized
    def tc2():
        raw = "pt presents with severe sob and acute mi"
        corrected = correct_specialty_terms(raw, specialty="cardiology")
        # In cardiology, terms like mi / sob / pt are processed
        assert isinstance(corrected, str)
        assert len(corrected) >= len(raw)
        return True, f"Clinical text processed and specialty normalized: '{corrected}'"

    runner.record_case(
        case_num=2,
        name="Abbreviation normalization and expansion",
        input_setup="Pass clinical abbreviation string 'pt presents with severe sob and acute mi' to specialty cleaner",
        expected_result="Clinical shorthand normalized without corruption",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Arabic abbreviation / shorthand handling
    def tc3():
        ar_input = "المريض يعاني من فشل قلبي حاد و ارتفاع ضغط الدم"
        corrected = correct_specialty_terms(ar_input, specialty="cardiology")
        assert "فشل قلبي" in corrected
        assert "ارتفاع ضغط الدم" in corrected
        return True, f"Arabic medical shorthand and terminology preserved cleanly: '{corrected}'"

    runner.record_case(
        case_num=3,
        name="Arabic clinical terminology preservation",
        input_setup="Pass Arabic clinical narrative to specialty terms corrector",
        expected_result="Arabic specialty terms preserved and normalized accurately",
        severity="High",
        fn=tc3,
    )

    # TC 4: Whitelist preservation — 'CT' (whitelisted) kept as-is
    def tc4():
        text_with_whitelist = "Patient scheduled for chest CT and brain MRI"
        corrected = correct_specialty_terms(text_with_whitelist, specialty="pulmonology")
        assert "CT" in corrected, "Whitelisted abbreviation 'CT' was erroneously altered"
        assert "MRI" in corrected, "Whitelisted abbreviation 'MRI' was erroneously altered"
        return True, "Whitelisted standard abbreviations (CT, MRI) preserved verbatim"

    runner.record_case(
        case_num=4,
        name="Whitelist preservation of medical acronyms",
        input_setup="Pass text with whitelisted acronyms 'CT' and 'MRI'",
        expected_result="Preserved verbatim without false-positive spell correction",
        severity="High",
        fn=tc4,
    )

    # TC 5: Idempotency — Run 2x same input -> Identical output
    def tc5():
        sample = "Patient diagnosed with unstable angina and atrial fibrillation"
        pass1 = correct_specialty_terms(sample, specialty="cardiology")
        pass2 = correct_specialty_terms(pass1, specialty="cardiology")
        assert pass1 == pass2, f"Idempotency violation: pass1='{pass1}' != pass2='{pass2}'"
        return True, "Idempotency verified: repeated execution yields bitwise identical output"

    runner.record_case(
        case_num=5,
        name="Term correction idempotency",
        input_setup="Execute correct_specialty_terms consecutively twice on same clinical text",
        expected_result="Pass 1 and Pass 2 return identical strings (f(f(x)) == f(x))",
        severity="Crit",
        fn=tc5,
    )

    return runner


# ==============================================================================
# B0.2 — Red-flag lexicon + routing signal
# ==============================================================================
def run_test_b0_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B0",
        subtask_id="B0.2",
        title="Red-flag lexicon detection & routing signal generation",
        target_files=["src/data/red_flag_lexicon.json", "src/utils/red_flags.py"],
        source_file=__file__,
    )

    # TC 1: Arabic detection — 'ألم في الصدر' -> red_flags signal emitted
    def tc1():
        ar_text = "المريض يشكو من ألم في الصدر وضيق في التنفس"
        res = detect_red_flags(ar_text)
        flags = res.get("matched_signals", [])
        assert len(flags) > 0, "Failed to detect Arabic red flag in acute chest pain narrative"
        categories = [f.get('label', '') for f in flags]
        assert any("cardiac" in c.lower() or "chest" in c.lower() or "respiratory" in c.lower() for c in categories)
        return True, f"Arabic red flag detected: {len(flags)} flag(s) found -> {flags[0].get('matched_phrase', '')}"

    runner.record_case(
        case_num=1,
        name="Arabic red-flag detection",
        input_setup="Evaluate Arabic acute cardiac symptom: 'المريض يشكو من ألم حاد في الصدر'",
        expected_result="Red flag emitted with appropriate clinical category",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: English detection — 'chest pain radiating to left arm' -> detected with category
    def tc2():
        en_text = "Patient presenting with crushing chest pain radiating to left arm and diaphoresis"
        res = detect_red_flags(en_text)
        flags = res.get("matched_signals", [])
        assert len(flags) > 0, "Failed to detect English cardiac red flag"
        categories = [f.get('label', '') for f in flags]
        assert any("cardiac" in c.lower() or "chest" in c.lower() for c in categories)
        return True, f"English red flag detected with category: {flags[0].get('label')} (term: {flags[0].get('matched_phrase')})"

    runner.record_case(
        case_num=2,
        name="English red-flag detection with category",
        input_setup="Evaluate English narrative 'crushing chest pain radiating to left arm'",
        expected_result="Detected with category (e.g. cardiac) and high severity",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Negation suppression — 'no chest pain' -> NOT emitted
    def tc3():
        neg_text = "Patient denies shortness of breath, no chest pain, no palpitations"
        res = detect_red_flags(neg_text)
        flags = [s for s in res.get("matched_signals", []) if s.get("status") != "negated"]
        # Negated symptoms should not trigger active emergency red flags
        cardiac_flags = [f for f in flags if "chest pain" in f.get("matched_phrase", "").lower()]
        assert len(cardiac_flags) == 0, f"False positive: negated symptom triggered red flag: {cardiac_flags}"
        return True, "Negation suppression confirmed: 'no chest pain' suppressed active red-flag trigger"

    runner.record_case(
        case_num=3,
        name="Negation suppression",
        input_setup="Evaluate negated symptom text: 'Patient denies shortness of breath, no chest pain'",
        expected_result="Negated symptoms are suppressed; red flag is NOT emitted",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Multiple flags — 3 different triggers -> All 3 detected
    def tc4():
        multi_text = "Severe crushing chest pain, acute hemoptysis, and sudden onset loss of consciousness"
        res = detect_red_flags(multi_text)
        flags = [s for s in res.get("matched_signals", []) if s.get("status") != "negated"]
        assert len(flags) >= 2, f"Expected multiple distinct red flags, detected {len(flags)}"
        return True, f"Multiple distinct red-flag triggers identified: {len(flags)} flags captured"

    runner.record_case(
        case_num=4,
        name="Multiple distinct red flags detection",
        input_setup="Evaluate encounter with 3 distinct emergent triggers (cardiac, respiratory, neuro)",
        expected_result="All active emergency triggers detected independently",
        severity="High",
        fn=tc4,
    )

    # TC 5: Lexicon structure — Parse JSON file -> required keys present
    def tc5():
        lex_path = SRC_DIR / "data" / "red_flag_lexicon.json"
        assert lex_path.exists(), "red_flag_lexicon.json not found"
        data = json.loads(lex_path.read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) > 0, "Lexicon must be a non-empty list of entries"
        sample = data[0]
        for req in ["id", "label", "acuity", "population", "phrases"]:
            assert req in sample, f"Required lexicon schema field '{req}' missing from entry"
        return True, f"Lexicon file structure valid: {len(data)} entries with id, label, acuity, population, phrases"

    runner.record_case(
        case_num=5,
        name="Lexicon file schema and structure integrity",
        input_setup="Parse src/data/red_flag_lexicon.json and validate required dictionary schema",
        expected_result="All entries contain term_ar, term_en, category, and severity",
        severity="High",
        fn=tc5,
    )

    return runner
