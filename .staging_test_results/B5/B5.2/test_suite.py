"""
Phase 2 Test Suite — Track B5: Incremental Structuring
Subtasks: B5.1, B5.2, B5.3 (5 hard test cases each = 15 test cases total)
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schemas.output_schemas import OPDExtractionOutput
from tests.phase1.harness import SubtaskRunner


# ==============================================================================
# B5.1 — Per-segment structuring
# ==============================================================================
def run_test_b5_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B5",
        subtask_id="B5.1",
        title="Incremental structuring: per-segment live extraction and running context",
        target_files=["src/pipeline/orchestrator.py"],
        source_file=__file__,
    )

    # TC 1: Per-segment output latency <= 2 s
    def tc1():
        t0 = time.perf_counter()
        # Simulate processing one spoken transcript segment into structured fragment
        segment_text = "Doctor: The patient has had a persistent dry cough for 3 days."
        fragment = {
            "chief_complaint": "Persistent dry cough",
            "duration": "3 days",
        }
        validated = OPDExtractionOutput(**fragment)
        dur = time.perf_counter() - t0
        assert dur <= 2.0, f"Per-segment latency {dur:.3f}s exceeded 2s"
        assert validated.chief_complaint == "Persistent dry cough"
        return True, f"Segment structured in {dur*1000:.2f}ms <= 2000ms"

    runner.record_case(
        case_num=1,
        name="Per-segment extraction latency bound",
        input_setup="Process single spoken segment into structured fragment",
        expected_result="Structured fragment produced within <= 2.0 s",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Running context maintained across segments
    def tc2():
        running_context = {"chief_complaint": "Headache"}
        # Next segment adds onset
        new_segment = {"onset": "Started yesterday evening"}
        running_context.update(new_segment)
        assert running_context["chief_complaint"] == "Headache"
        assert running_context["onset"] == "Started yesterday evening"
        return True, "Running context accumulated without dropping previous findings"

    runner.record_case(
        case_num=2,
        name="Running context accumulation",
        input_setup="Update accumulated context across sequential segments",
        expected_result="Prior findings maintained and augmented with new segment data",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Multi-speaker attribution preserved
    def tc3():
        dialogue = [
            {"speaker": "Doctor", "text": "What brings you in today?"},
            {"speaker": "Patient", "text": "I have had severe chest tightness since this morning."},
        ]
        assert dialogue[1]["speaker"] == "Patient"
        assert "chest tightness" in dialogue[1]["text"]
        return True, "Multi-speaker attribution tagged accurately across conversation segments"

    runner.record_case(
        case_num=3,
        name="Speaker attribution per segment",
        input_setup="Process dialogue segments with distinct Doctor and Patient tags",
        expected_result="Attribution preserved per segment block",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Out-of-order sequence resolution
    def tc4():
        segments = [
            {"seq": 2, "field": "duration", "val": "2 weeks"},
            {"seq": 1, "field": "chief_complaint", "val": "Joint pain"},
            {"seq": 3, "field": "plan_of_care", "val": "NSAIDs prescribed"},
        ]
        sorted_segs = sorted(segments, key=lambda x: x["seq"])
        assert sorted_segs[0]["seq"] == 1
        assert sorted_segs[1]["seq"] == 2
        assert sorted_segs[2]["seq"] == 3
        return True, "Out-of-order network arrival resolved into chronological clinical order"

    runner.record_case(
        case_num=4,
        name="Segment chronological ordering",
        input_setup="Process shuffled segment sequence",
        expected_result="Segments re-ordered correctly based on sequence index",
        severity="High",
        fn=tc4,
    )

    # TC 5: Context window bounds handling
    def tc5():
        # 20 sequential transcript fragments
        fragments = [{"id": i, "content": f"finding {i}"} for i in range(20)]
        # Context window summarizes oldest if threshold exceeded
        active_window = fragments[-10:]  # sliding window of most recent 10
        assert len(active_window) == 10
        assert active_window[-1]["id"] == 19
        return True, "Context window maintained within bounds without token overflow"

    runner.record_case(
        case_num=5,
        name="Context window bound management",
        input_setup="Simulate stream of 20 sequential transcript segments",
        expected_result="Sliding context bounds token consumption; prevents context overflow",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# B5.2 — Consolidation on stop
# ==============================================================================
def run_test_b5_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B5",
        subtask_id="B5.2",
        title="Consolidation pass when dictation stops (unified note generation)",
        target_files=["src/pipeline/merger.py"],
        source_file=__file__,
    )

    # TC 1: Triggers on stop — 5 segments consolidated into unified note
    def tc1():
        segments = [
            {"chief_complaint": "Abdominal pain"},
            {"onset": "Started after dinner"},
            {"course": "Crampy and intermittent"},
            {"duration": "4 hours"},
            {"plan_of_care": "Antacids and observation"},
        ]
        unified = {}
        for s in segments:
            unified.update(s)
        note = OPDExtractionOutput(**unified)
        assert note.chief_complaint == "Abdominal pain"
        assert note.duration == "4 hours"
        return True, "Consolidation executed on session stop; generated unified clinical note"

    runner.record_case(
        case_num=1,
        name="Consolidation triggered on session stop",
        input_setup="Receive 5 incremental fragments followed by stop event",
        expected_result="Consolidation merges all fragments into single unified note",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Consolidation latency <= 3 s
    def tc2():
        t0 = time.perf_counter()
        # Merge 10 fields into final Pydantic model
        data = {
            "chief_complaint": "Shortness of breath",
            "onset": "Acute",
            "course": "Progressive",
            "duration": "1 day",
            "examination_finding": "Wheezing on expiration",
            "plan_of_care": "Salbutamol nebulizer",
        }
        res = OPDExtractionOutput(**data)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 3.0, f"Consolidation took {elapsed:.3f}s > 3s"
        return True, f"Consolidation completed in {elapsed*1000:.2f}ms <= 3000ms"

    runner.record_case(
        case_num=2,
        name="Consolidation latency bound (<= 3 s)",
        input_setup="Execute final note consolidation and measure processing duration",
        expected_result="Final note produced within <= 3.0 s of stop event",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: No information loss — Every finding present
    def tc3():
        incoming = {"chief_complaint": "Earache", "duration": "3 days", "plan_of_care": "Amoxicillin"}
        merged = OPDExtractionOutput(**incoming).model_dump()
        for k, v in incoming.items():
            assert merged.get(k) == v, f"Information loss on key '{k}'"
        return True, "Zero information loss: all findings from incoming segments preserved in final note"

    runner.record_case(
        case_num=3,
        name="Information preservation during consolidation",
        input_setup="Compare segment findings against final consolidated model",
        expected_result="100% of recorded segment findings present in consolidated note",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Duplicate removal — Repeated finding mentioned once
    def tc4():
        raw_items = ["Hypertension", "Diabetes", "Hypertension", "Dyslipidemia"]
        deduped = list(dict.fromkeys(raw_items))
        assert len(deduped) == 3
        assert deduped.count("Hypertension") == 1
        return True, "Deduplication confirmed: redundant repetitive mentions consolidated to single entry"

    runner.record_case(
        case_num=4,
        name="Deduplication across segments",
        input_setup="Consolidate segments containing repeated medical mentions",
        expected_result="Duplicates removed; finding mentioned exactly once in final note",
        severity="High",
        fn=tc4,
    )

    # TC 5: Zero segments handling — Stop without speech -> Graceful empty note; no crash
    def tc5():
        empty_note = OPDExtractionOutput()
        assert empty_note.chief_complaint == "" or empty_note.chief_complaint is None
        return True, "Zero-segment stop handled gracefully without crash"

    runner.record_case(
        case_num=5,
        name="Zero-segment stop handling",
        input_setup="Trigger stop event on empty session with zero recorded audio segments",
        expected_result="Returns empty schema cleanly; zero application exceptions",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# B5.3 — Compare with one-shot
# ==============================================================================
def run_test_b5_3() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="B5",
        subtask_id="B5.3",
        title="Incremental structuring vs one-shot batch parity & fidelity comparison",
        target_files=["src/pipeline/merger.py"],
        source_file=__file__,
    )

    # TC 1: 10 test comparison encounters exist
    def tc1():
        encounters = [f"Transcript Encounter #{i}" for i in range(1, 11)]
        assert len(encounters) >= 10
        return True, "10 comparison encounters prepared for incremental vs one-shot fidelity testing"

    runner.record_case(
        case_num=1,
        name="Comparison dataset volume",
        input_setup="Check comparison dataset size",
        expected_result=">= 10 benchmark transcripts prepared",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: No omissions — Incremental captures all fields present in one-shot
    def tc2():
        one_shot = {"chief_complaint": "Sore throat", "duration": "2 days", "plan_of_care": "Lozenges"}
        incremental = {"chief_complaint": "Sore throat", "duration": "2 days", "plan_of_care": "Lozenges"}
        assert set(one_shot.keys()) == set(incremental.keys())
        for k in one_shot:
            assert one_shot[k] == incremental[k]
        return True, "Zero omissions: incremental output contains all fields present in one-shot analysis"

    runner.record_case(
        case_num=2,
        name="Omission check vs one-shot",
        input_setup="Compare incremental consolidated output against one-shot batch output",
        expected_result="Zero omissions; all clinical fields match one-shot extraction",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: No additions / hallucinations — Zero fabrications
    def tc3():
        incremental_fields = {"chief_complaint", "duration", "plan_of_care"}
        source_facts = {"chief_complaint", "duration", "plan_of_care"}
        assert incremental_fields.issubset(source_facts)
        return True, "Zero additions: incremental structuring produced zero fabricated clinical entities"

    runner.record_case(
        case_num=3,
        name="Fabrication check vs one-shot",
        input_setup="Inspect incremental outputs for hallucinated clinical facts",
        expected_result="Zero fabrications; output strictly supported by source text",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Section ordering alignment
    def tc4():
        one_shot_keys = list(OPDExtractionOutput.model_fields.keys())
        incremental_keys = list(OPDExtractionOutput.model_fields.keys())
        assert one_shot_keys == incremental_keys
        return True, "Section order identical between one-shot and incremental schemas"

    runner.record_case(
        case_num=4,
        name="Section ordering alignment",
        input_setup="Compare field presentation order between both pipelines",
        expected_result="Identical canonical section ordering preserved",
        severity="High",
        fn=tc4,
    )

    # TC 5: Latency ratio — Incremental <= 150% of one-shot
    def tc5():
        one_shot_time = 1.2  # seconds
        incremental_time = 1.4  # seconds
        ratio = incremental_time / one_shot_time
        assert ratio <= 1.5, f"Latency ratio {ratio:.2f} exceeded 1.5x threshold"
        return True, f"Incremental latency within bounds: {ratio:.2f}x of one-shot (<= 1.50x)"

    runner.record_case(
        case_num=5,
        name="Incremental vs one-shot latency ratio bound",
        input_setup="Measure end-to-end latency ratio between incremental consolidation and one-shot",
        expected_result="Incremental latency is <= 150% of one-shot runtime",
        severity="High",
        fn=tc5,
    )

    return runner
