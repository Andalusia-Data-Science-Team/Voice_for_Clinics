"""
MedVoice v3.4 — Phase 3 Master Verification Orchestrator
Φ-3: Orchestration & Integration Verification Runner

Tracks:
  - M4: Parallel Orchestration (M4.1, M4.2, M4.3)
  - M6: Carry Forward (M6.1, M6.2, M6.3)
  - M7: Auth & Privacy (M7.1, M7.2)
  - A4: CI/CD Pipeline (A4.1, A4.2, A4.3)
  - A5: Load Testing & Telemetry (A5.1, A5.2)
  - A7: Reliability & Hardening (A7.1, A7.2, A7.3, A7.4)

Total: 17 subtasks x 5 hard test cases = 85 tests.
Quality Gate: 100% pass (85/85) required to clear Phase 3.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.phase3.test_m4 import run_test_m4_1, run_test_m4_2, run_test_m4_3
from tests.phase3.test_m6 import run_test_m6_1, run_test_m6_2, run_test_m6_3
from tests.phase3.test_m7 import run_test_m7_1, run_test_m7_2
from tests.phase3.test_a4 import run_test_a4_1, run_test_a4_2, run_test_a4_3
from tests.phase3.test_a5 import run_test_a5_1, run_test_a5_2
from tests.phase3.test_a7 import (
    run_test_a7_1,
    run_test_a7_2,
    run_test_a7_3,
    run_test_a7_4,
)

ALL_SUBTASK_FACTORIES = [
    # M4
    ("M4", "M4.1", run_test_m4_1),
    ("M4", "M4.2", run_test_m4_2),
    ("M4", "M4.3", run_test_m4_3),
    # M6
    ("M6", "M6.1", run_test_m6_1),
    ("M6", "M6.2", run_test_m6_2),
    ("M6", "M6.3", run_test_m6_3),
    # M7
    ("M7", "M7.1", run_test_m7_1),
    ("M7", "M7.2", run_test_m7_2),
    # A4
    ("A4", "A4.1", run_test_a4_1),
    ("A4", "A4.2", run_test_a4_2),
    ("A4", "A4.3", run_test_a4_3),
    # A5
    ("A5", "A5.1", run_test_a5_1),
    ("A5", "A5.2", run_test_a5_2),
    # A7
    ("A7", "A7.1", run_test_a7_1),
    ("A7", "A7.2", run_test_a7_2),
    ("A7", "A7.3", run_test_a7_3),
    ("A7", "A7.4", run_test_a7_4),
]


def run_phase3_suite() -> int:
    start_time = time.time()
    print("=" * 80)
    print(" MedVoice v3.4 — Phase 3 (Φ-3) Master Verification Suite")
    print(" Tracks: M4, M6, M7, A4, A5, A7 (17 subtasks | 85 rigorous test cases)")
    print("=" * 80)

    results = []
    for task_id, subtask_id, factory in ALL_SUBTASK_FACTORIES:
        print(f"\n>> Executing verification runner for {task_id}.{subtask_id}...")
        runner = factory()
        res = runner.finalize()
        results.append((task_id, subtask_id, res))

    total_subtasks = len(results)
    cleared_subtasks = sum(1 for _, _, r in results if r.passed)
    total_cases = sum(len(r.cases) for _, _, r in results)
    passed_cases = sum(r.pass_count for _, _, r in results)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(" Phase 3 Verification Results Summary")
    print("=" * 80)
    for task_id, subtask_id, res in results:
        status_str = "CLEARED" if res.passed else "FAILED"
        rate = (res.pass_count / len(res.cases) * 100) if res.cases else 0.0
        print(f"  [{status_str:7s}] {task_id}.{subtask_id:4s} — {res.pass_count}/{len(res.cases)} tests passed ({rate:.1f}%)")

    print("-" * 80)
    print(f"Subtasks Cleared : {cleared_subtasks}/{total_subtasks} ({cleared_subtasks/total_subtasks*100:.1f}%)")
    print(f"Test Cases Passed: {passed_cases}/{total_cases} ({passed_cases/total_cases*100:.1f}%)")
    print(f"Duration         : {elapsed:.2f}s")
    print("=" * 80)

    if cleared_subtasks == total_subtasks and passed_cases == total_cases:
        print("\n>>> ALL PHASE 3 SUBTASKS MET QUALITY GATE (85/85 PASSED). PROMOTION UNLOCKED. <<<\n")
        return 0
    else:
        print("\n>>> PHASE 3 QUALITY GATE FAILED. FIX DEFICIENCIES BEFORE PROCEEDING. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_phase3_suite())
