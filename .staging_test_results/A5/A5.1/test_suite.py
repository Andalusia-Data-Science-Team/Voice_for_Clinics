"""
Phase 3 Test Suite — Track A5: Performance & Load Testing
Subtasks: A5.1 (Locust script structure & realism), A5.2 (Export session timings)
(5 hard test cases each = 10 total)
"""

from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.phase1.harness import SubtaskRunner


# ==============================================================================
# A5.1 — Locust script structure & realism
# ==============================================================================
def run_test_a5_1() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A5",
        subtask_id="A5.1",
        title="Locust load testing script imports cleanly and defines realistic workload patterns",
        target_files=["locustfile.py"],
        source_file=__file__,
    )

    # TC 1: locustfile imports cleanly without error
    def tc1():
        import importlib
        try:
            import locustfile
            return True, "locustfile imported cleanly without exception"
        except Exception as e:
            return False, f"Failed to import locustfile: {e}"

    runner.record_case(
        case_num=1,
        name="locustfile imports cleanly",
        input_setup="import locustfile in Python environment with locust installed",
        expected_result="locustfile imports with zero errors or unhandled exceptions",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: MedVoiceAnalyzeUser defines realistic waiting and pacing
    def tc2():
        import locustfile
        cls = getattr(locustfile, "MedVoiceAnalyzeUser", None)
        assert cls is not None, "MedVoiceAnalyzeUser not defined in locustfile"
        assert hasattr(cls, "wait_time"), "MedVoiceAnalyzeUser has no wait_time"
        # Has task decorated methods
        tasks = getattr(cls, "tasks", [])
        assert len(tasks) >= 1 or hasattr(cls, "analyze_transcript"), "No analyze task found on user"
        return True, "MedVoiceAnalyzeUser defines wait_time pacing and analyze tasks"

    runner.record_case(
        case_num=2,
        name="User model defines realistic pacing (wait_time)",
        input_setup="Inspect MedVoiceAnalyzeUser attributes and tasks",
        expected_result="wait_time and analyze tasks present on class",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Multi-language and mode mixtures configured
    def tc3():
        import locustfile
        cls = locustfile.MedVoiceAnalyzeUser
        assert hasattr(cls, "language_names") and hasattr(cls, "language_weights"), "Missing language mix"
        assert hasattr(cls, "mode_names") and hasattr(cls, "mode_weights"), "Missing mode mix"
        assert set(cls.language_names).issubset({"en", "ar"}), f"Unexpected languages: {cls.language_names}"
        assert set(cls.mode_names).issubset({"doctor", "conversation"}), f"Unexpected modes: {cls.mode_names}"
        return True, f"Mixtures verified: langs={cls.language_names}, modes={cls.mode_names}"

    runner.record_case(
        case_num=3,
        name="Realistic language and clinical mode distributions",
        input_setup="Check language_names, mode_names, and weights on MedVoiceAnalyzeUser",
        expected_result="Both EN/AR languages and doctor/conversation modes supported with weights",
        severity="High",
        fn=tc3,
    )

    # TC 4: Synthetic dataset presence for both English and Arabic
    def tc4():
        import locustfile
        syn_trans = getattr(locustfile, "SYNTHETIC_TRANSCRIPTS", None)
        assert syn_trans is not None, "SYNTHETIC_TRANSCRIPTS missing"
        assert "en" in syn_trans and "ar" in syn_trans, "Missing en or ar in synthetic transcripts"
        for lang in ("en", "ar"):
            for length in ("short", "medium", "long"):
                assert length in syn_trans[lang], f"Missing length {length} for {lang}"
                assert len(syn_trans[lang][length]) > 10, f"Transcript too short for {lang}:{length}"
        return True, "Complete synthetic clinical transcripts present for en/ar across short/medium/long"

    runner.record_case(
        case_num=4,
        name="Synthetic clinical transcripts available for English and Arabic",
        input_setup="Inspect SYNTHETIC_TRANSCRIPTS dictionary across languages and length buckets",
        expected_result="short, medium, long lengths present for both 'en' and 'ar'",
        severity="High",
        fn=tc4,
    )

    # TC 5: Failure category classifier prevents masked errors
    def tc5():
        import locustfile
        import requests.exceptions
        classifier = getattr(locustfile, "_failure_category", None)
        assert callable(classifier), "_failure_category function missing"
        assert classifier(requests.exceptions.Timeout("request timed out")) == "Timeout"
        assert classifier(requests.exceptions.ConnectionError("refused")) == "Connection error"
        assert classifier(RuntimeError("generic")) == "Stream error"
        return True, "_failure_category correctly tags Timeout, Connection error, and Stream error"

    runner.record_case(
        case_num=5,
        name="Granular failure classification helper",
        input_setup="Call _failure_category with simulated TimeoutError and ConnectionError",
        expected_result="Accurate semantic strings returned without swallowing root cause",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# A5.2 — Export session timings
# ==============================================================================
def run_test_a5_2() -> SubtaskRunner:
    runner = SubtaskRunner(
        task_id="A5",
        subtask_id="A5.2",
        title="Session CSV export captures timings: first partial, final transcript, first SSE, total",
        target_files=["locustfile.py"],
        source_file=__file__,
    )

    # TC 1: CSV target and tagging mechanism
    def tc1():
        import locustfile
        env = MagicMock()
        tag = locustfile._session_csv_tag(env)
        assert isinstance(tag, str) and len(tag) > 0, "Empty run tag"
        target = locustfile._session_csv_target(env)
        assert isinstance(target, str) and target.endswith(".csv"), f"Invalid CSV target path: {target}"
        return True, f"Session CSV tagging and target resolution verified (target={target})"

    runner.record_case(
        case_num=1,
        name="Session CSV tag and path resolution",
        input_setup="Call _session_csv_tag and _session_csv_target with mock environment",
        expected_result="Deterministic target string ending in .csv",
        severity="Crit",
        fn=tc1,
    )

    # TC 2: Required columns present in _write_session_row definition
    def tc2():
        import inspect
        import locustfile
        src = inspect.getsource(locustfile._write_session_row)
        required_cols = [
            "session_id", "language", "mode",
            "first_partial", "final_transcript", "first_sse_event", "total_sec"
        ]
        for col in required_cols:
            assert f'"{col}"' in src or f"'{col}'" in src, f"Column '{col}' missing from _write_session_row fields"
        return True, f"All required telemetry columns verified in writer: {required_cols}"

    runner.record_case(
        case_num=2,
        name="All required timing columns present in CSV writer schema",
        input_setup="Inspect _write_session_row fields definition in locustfile.py",
        expected_result="Fields list contains session_id, language, mode, first_partial, final_transcript, first_sse_event, total_sec",
        severity="Crit",
        fn=tc2,
    )

    # TC 3: Actual row writing produces well-formed CSV
    def tc3():
        import locustfile
        temp_csv = ROOT_DIR / "tests" / "phase3" / "scratch_locust_test.csv"
        old_target = locustfile._session_csv_target_path
        old_header = locustfile._session_csv_header_written
        try:
            locustfile._session_csv_target_path = str(temp_csv)
            locustfile._session_csv_header_written = False
            if temp_csv.exists():
                temp_csv.unlink()

            env = MagicMock()
            row_data = {
                "session_id": "sess-test-123",
                "language": "ar",
                "mode": "doctor",
                "first_partial": "المريض",
                "final_transcript": "المريض يعاني من سعال",
                "first_sse_event": "step_progress",
                "total_sec": 4.125,
            }
            locustfile._write_session_row(env, row_data)

            assert temp_csv.exists(), "CSV file was not created"
            with open(temp_csv, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                assert len(reader) == 1, f"Expected 1 data row, got {len(reader)}"
                row = reader[0]
                assert row["session_id"] == "sess-test-123"
                assert row["language"] == "ar"
                assert row["total_sec"] == "4.125"
                assert row["first_sse_event"] == "step_progress"
            return True, "CSV file created, header written, row appended with valid contents"
        finally:
            if temp_csv.exists():
                temp_csv.unlink()
            locustfile._session_csv_target_path = old_target
            locustfile._session_csv_header_written = old_header

    runner.record_case(
        case_num=3,
        name="Row writing outputs parseable CSV with correct values",
        input_setup="Directly execute _write_session_row to a temporary CSV file",
        expected_result="Valid CSV header and data row created with matching field values",
        severity="Crit",
        fn=tc3,
    )

    # TC 4: Metric firing helper _fire_sse_metric does not throw
    def tc4():
        import locustfile
        env = MagicMock()
        env.events.request.fire = MagicMock()
        locustfile._fire_sse_metric(env, "Test metric", 1250.0)
        assert env.events.request.fire.called, "request.fire was not called"
        return True, "_fire_sse_metric cleanly dispatches metric to Locust events"

    runner.record_case(
        case_num=4,
        name="_fire_sse_metric correctly dispatches custom metric",
        input_setup="Call _fire_sse_metric with mock Environment and custom metric name",
        expected_result="events.request.fire called without exception",
        severity="High",
        fn=tc4,
    )

    # TC 5: Threshold failure evaluation hook
    def tc5():
        import locustfile
        env = MagicMock()
        mock_entry = MagicMock()
        mock_entry.num_requests = 100
        mock_entry.num_failures = 10
        mock_entry.get_response_time_percentile.return_value = 2500
        mock_entry.percentile.return_value = 2500

        env.stats.get.return_value = mock_entry
        env.process_exit_code = 0

        orig_ratio = os.environ.get("LOCUST_MAX_FAIL_RATIO")
        try:
            os.environ["LOCUST_MAX_FAIL_RATIO"] = "0.01"
            locustfile._check_thresholds(env)
            assert env.process_exit_code == 1, "Exit code 1 was not set for threshold violation"
            return True, "_check_thresholds sets process_exit_code=1 on fail ratio violation"
        finally:
            if orig_ratio is not None:
                os.environ["LOCUST_MAX_FAIL_RATIO"] = orig_ratio
            else:
                os.environ.pop("LOCUST_MAX_FAIL_RATIO", None)

    runner.record_case(
        case_num=5,
        name="Automated threshold evaluation triggers on SLA violation",
        input_setup="Configure mock environment exceeding LOCUST_MAX_FAIL_RATIO",
        expected_result="process_exit_code set to 1 indicating automated test failure",
        severity="High",
        fn=tc5,
    )

    return runner


# ==============================================================================
# Main runner
# ==============================================================================
if __name__ == "__main__":
    results = []

    r = run_test_a5_1()
    res = r.finalize()
    results.append(res)

    r = run_test_a5_2()
    res = r.finalize()
    results.append(res)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"A5 TOTAL: {passed}/{total} subtasks gate-cleared")
