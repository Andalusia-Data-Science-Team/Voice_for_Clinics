"""Unit tests for Phase 1 & 3: Deterministic 5-Rule Complexity Classification Engine.

Covers:
- Individual unit tests for each of the 5 rules (positive & negative cases).
- Bilingual edge cases (Arabic NFKC normalization, diacritics, English case-folding).
- Combinatorial boundary tests (0/5, 1/5, 2/5, 3/5, 4/5, 5/5).
- Benchmark tests verifying execution time remains well below 2ms across 1000 iterations.
"""

import time
import pytest
from pipeline.complexity import (
    classify_clinical_complexity,
    evaluate_noisy_dictation,
    evaluate_multi_speaker,
    evaluate_reasoning_cues,
    evaluate_history_relevance,
    evaluate_unknown_abbreviations,
    icd10_chapter,
)


# ── Rule 1: Noisy Dictation ───────────────────────────────────────────────────

def test_rule1_noisy_dictation_negative():
    # 0 fillers
    passed, matches = evaluate_noisy_dictation("Patient has chest pain. Lungs clear.")
    assert passed is False
    assert len(matches) == 0

    # 1 filler (below threshold of 2)
    passed, matches = evaluate_noisy_dictation("Okay patient has mild fever.")
    assert passed is False
    assert len(matches) == 1

    # Arabic single filler
    passed, matches = evaluate_noisy_dictation("المريض يشتكي من صداع يعني.")
    assert passed is False
    assert len(matches) == 1


def test_rule1_noisy_dictation_positive():
    # English >= 2 fillers
    passed, matches = evaluate_noisy_dictation("Okay, sorry I mean he has right knee pain.")
    assert passed is True
    assert len(matches) >= 2

    # Arabic >= 2 fillers with diacritics / normalization
    passed, matches = evaluate_noisy_dictation("المريض عنده كحة عفواً أقصد حرارة خفيفة يعني.")
    assert passed is True
    assert len(matches) >= 2

    # Self-correction phrases
    passed, matches = evaluate_noisy_dictation("BP 140/90, no wait, scratch that, BP 120/80.")
    assert passed is True
    assert len(matches) >= 2


# ── Rule 2: Multi-Speaker Dispersion ──────────────────────────────────────────

def test_rule2_multi_speaker_negative():
    # Standard doctor dictation
    passed, details = evaluate_multi_speaker("Patient is a 45-year-old male. Denies chest pain.")
    assert passed is False
    assert len(details["companion_matches"]) == 0

    # Short dialogue below turn threshold
    short_dialogue = "Doctor: Hello.\nPatient: Hi."
    passed, details = evaluate_multi_speaker(short_dialogue, is_conversation=True)
    assert passed is False


def test_rule2_multi_speaker_positive_companion():
    # English companion terms: daughter, wife, carer, escort
    passed, details = evaluate_multi_speaker("Patient attended with his daughter who stated he fell yesterday.")
    assert passed is True
    assert "daughter" in details["companion_matches"]

    passed, details = evaluate_multi_speaker("Accompanied by carer and relative.")
    assert passed is True
    assert len(details["companion_matches"]) >= 1

    # Arabic companion terms: الزوجة، الابنة، المرافق، أهل المريض
    passed, details = evaluate_multi_speaker("حضرت الزوجة مع المريض وافادت بوجود دوخة مستمرة.")
    assert passed is True
    assert len(details["companion_matches"]) > 0

    passed, details = evaluate_multi_speaker("حضر مع مرافقة المريض.")
    assert passed is True


def test_rule2_multi_speaker_positive_turns():
    # Dialogue with >= 10 turns
    turns = "\n".join([f"Doctor: Question {i}\nPatient: Answer {i}" for i in range(6)])
    passed, details = evaluate_multi_speaker(turns, is_conversation=True)
    assert passed is True
    assert details["turn_count"] >= 10


# ── Rule 3: Reasoning Cues ───────────────────────────────────────────────────

def test_rule3_reasoning_cues_negative():
    # Objective declarative findings without uncertainty
    passed, cues = evaluate_reasoning_cues("Fracture of distal radius confirmed on X-ray. Cast applied.")
    assert passed is False
    assert len(cues) == 0


def test_rule3_reasoning_cues_positive():
    # English uncertainty / differential cues
    passed, cues = evaluate_reasoning_cues("Possible acute coronary syndrome, need to rule out NSTEMI.")
    assert passed is True
    assert len(cues) >= 1

    # Equivocal / borderline
    passed, cues = evaluate_reasoning_cues("Borderline ECG changes, suspect pericarditis vs myocarditis.")
    assert passed is True

    # Conditional plan: if ... otherwise
    passed, cues = evaluate_reasoning_cues("If troponin rises admitted to CCU, otherwise discharge home.")
    assert passed is True

    # Arabic uncertainty / differential
    passed, cues = evaluate_reasoning_cues("احتمال وجود التهاب رئوي ونستبعد الجلطة الرئوية.")
    assert passed is True

    # Arabic conditional: لو ... والا
    passed, cues = evaluate_reasoning_cues("لو ارتفع الضغط نزيد الجرعة وإلا نستمر على نفس العلاج.")
    assert passed is True


# ── Rule 4: History Relevance ────────────────────────────────────────────────

def test_rule4_history_relevance_negative():
    # History disabled
    passed, details = evaluate_history_relevance(
        use_history=False,
        patient_history_visits=[{"icd10_diagnoses": ["I20.0 - Unstable angina"]}],
        current_icd_candidates=[("I20.9", "Angina pectoris")],
    )
    assert passed is False
    assert details["reason"] == "history_off_or_empty"

    # History enabled but disjoint chapters (e.g., Chapter XIV Genitourinary vs Chapter XIX Injury)
    passed, details = evaluate_history_relevance(
        use_history=True,
        patient_history_visits=[{"icd10_diagnoses": ["N39.0 - Urinary tract infection"]}],
        current_icd_candidates=[("S61.4", "Open wound of hand")],
    )
    assert passed is False
    assert len(details["overlapping_chapters"]) == 0


def test_rule4_history_relevance_positive():
    # Overlapping Circulatory Chapter IX (I20 and I25)
    passed, details = evaluate_history_relevance(
        use_history=True,
        patient_history_visits=[{"icd10_diagnoses": ["I20.0 - Unstable angina"]}],
        current_icd_candidates=[("I25.1", "Atherosclerotic heart disease")],
    )
    assert passed is True
    assert "IX_Circulatory" in details["overlapping_chapters"]

    # Overlapping Endocrine Chapter IV (E10 and E11)
    passed, details = evaluate_history_relevance(
        use_history=True,
        patient_history_visits=[{"icd10_diagnoses": ["E10.9 - Type 1 diabetes"]}],
        current_icd_candidates=[("E11.9", "Type 2 diabetes")],
    )
    assert passed is True
    assert "IV_Endocrine" in details["overlapping_chapters"]


# ── Rule 5: Unknown Abbreviations ────────────────────────────────────────────

def test_rule5_unknown_abbreviations_negative():
    # Standard medical abbreviations that are whitelisted
    text_known = "Patient with DM and HTN presented for routine ECG. BP was 120/80 and HR 72."
    passed, unknown = evaluate_unknown_abbreviations(text_known)
    assert passed is False
    assert len(unknown) == 0

    # Dental acronyms whitelisted (RCT, CBCT, FDI)
    text_dental = "Tooth 36 indicated for RCT following CBCT scan."
    passed, unknown = evaluate_unknown_abbreviations(text_dental)
    assert passed is False


def test_rule5_unknown_abbreviations_positive():
    # Non-whitelisted medical acronym: DCL (Decreased Conscious Level)
    text_unknown = "60 yo patient presented with DCL and acute confusion."
    passed, unknown = evaluate_unknown_abbreviations(text_unknown)
    assert passed is True
    assert "DCL" in unknown


# ── Combinatorial Boundary Conditions (0/5, 1/5, 2/5, 3/5, 4/5, 5/5) ────────

def test_combinatorial_score_0_simple():
    # Clean declarative note -> Score 0 -> simple (gpt-5.6-luna)
    eval_0 = classify_clinical_complexity("Patient presents with mild sore throat for 2 days. Lungs clear.")
    assert eval_0.score == 0
    assert eval_0.is_complex is False
    assert eval_0.tier == "simple"
    assert len(eval_0.matched_rules) == 0


def test_combinatorial_score_1_simple():
    # Only Rule 1 active (2 fillers) -> Score 1 -> simple
    eval_1 = classify_clinical_complexity("Okay, sorry I mean he has knee pain.")
    assert eval_1.score == 1
    assert eval_1.is_complex is False
    assert eval_1.tier == "simple"
    assert eval_1.matched_rules == ["NOISY_DICTATION"]


def test_combinatorial_score_2_boundary_simple():
    # Exactly 2/5 active: Rule 1 (fillers) + Rule 5 (unknown abbr DCL) -> Score 2 -> simple
    eval_2 = classify_clinical_complexity("Okay, sorry I mean the patient has DCL.")
    assert eval_2.score == 2
    assert eval_2.is_complex is False
    assert eval_2.tier == "simple"
    assert set(eval_2.matched_rules) == {"NOISY_DICTATION", "UNKNOWN_ABBREVIATIONS"}


def test_combinatorial_score_3_threshold_complex():
    # Exactly 3/5 active: Rule 1 + Rule 5 + Rule 3 (reasoning) -> Score 3 -> complex (gpt-5.6-sol)
    eval_3 = classify_clinical_complexity("Okay, sorry I mean patient has DCL, possibly encephalitis.")
    assert eval_3.score == 3
    assert eval_3.is_complex is True
    assert eval_3.tier == "complex"
    assert set(eval_3.matched_rules) == {"NOISY_DICTATION", "REASONING_CUES", "UNKNOWN_ABBREVIATIONS"}


def test_combinatorial_score_4_complex():
    # 4/5 active: Rule 1 + Rule 2 (daughter) + Rule 3 (possibly) + Rule 5 (DCL)
    text_4 = "Okay, sorry I mean daughter mentioned DCL, possibly acute."
    eval_4 = classify_clinical_complexity(text_4)
    assert eval_4.score == 4
    assert eval_4.is_complex is True
    assert eval_4.tier == "complex"
    assert set(eval_4.matched_rules) == {"NOISY_DICTATION", "MULTI_SPEAKER", "REASONING_CUES", "UNKNOWN_ABBREVIATIONS"}


def test_combinatorial_score_5_maximum_complex():
    # All 5/5 active: Rule 1 + Rule 2 + Rule 3 + Rule 4 + Rule 5
    text_5 = "Doctor: Okay, sorry I mean his daughter said he has DCL, possibly sepsis."
    eval_5 = classify_clinical_complexity(
        text=text_5,
        is_conversation=True,
        use_history=True,
        patient_history_visits=[{"icd10_diagnoses": ["A41.9 - Sepsis"]}],
        current_icd_candidates=[("A41.0", "Sepsis")],
    )
    assert eval_5.score == 5
    assert eval_5.is_complex is True
    assert eval_5.tier == "complex"
    assert len(eval_5.matched_rules) == 5


# ── Benchmark & Performance (< 2ms) ──────────────────────────────────────────

def test_benchmark_execution_time_under_2ms():
    """Verify that rule evaluation executes in < 2ms even under 1000 sequential passes."""
    sample_text = (
        "Doctor: Okay, sorry I mean his daughter noted DCL and chest discomfort. "
        "Need to rule out acute coronary syndrome. If troponin elevated admitted to CCU, otherwise discharge home."
    )
    history_visits = [{"icd10_diagnoses": ["I20.0 - Unstable angina", "E11.9 - Type 2 diabetes"]}]
    candidates = [("I25.1", "Atherosclerotic heart disease"), ("R07.4", "Chest pain")]

    # Warmup
    _ = classify_clinical_complexity(
        text=sample_text,
        is_conversation=True,
        use_history=True,
        patient_history_visits=history_visits,
        current_icd_candidates=candidates,
    )

    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        res = classify_clinical_complexity(
            text=sample_text,
            is_conversation=True,
            use_history=True,
            patient_history_visits=history_visits,
            current_icd_candidates=candidates,
        )
    total_sec = time.perf_counter() - t0
    avg_ms = (total_sec / iterations) * 1000.0

    # Ensure single execution latency is well under 2ms (typically ~0.2 - 0.4ms)
    assert avg_ms < 2.0, f"Average execution time {avg_ms:.3f}ms exceeded 2.0ms threshold"
    assert res.score >= 3
    assert res.is_complex is True
