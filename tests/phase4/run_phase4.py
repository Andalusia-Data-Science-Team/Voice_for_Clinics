"""
MedVoice v3.4 — Phase 4 Master Verification Orchestrator
Φ-4: UI, Regression & Performance Verification Runner

Tracks:
  - M5: UI Features (M5.1, M5.2, M5.3, M5.4)
  - M8: Functional Regression (M8.1, M8.2)
  - A6: Load Testing & Limits (A6.1, A6.2, A6.3)

Total: 9 subtasks x 5 hard test cases = 45 tests.
Quality Gate: 100% pass (45/45) required to clear Phase 4.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.phase4.test_m5 import run_test_m5_1, run_test_m5_2, run_test_m5_3, run_test_m5_4
from tests.phase4.test_m8 import run_test_m8_1, run_test_m8_2
from tests.phase4.test_a6 import run_test_a6_1, run_test_a6_2, run_test_a6_3

ALL_SUBTASK_FACTORIES = [
    # M5
    ("M5", "M5.1", run_test_m5_1),
    ("M5", "M5.2", run_test_m5_2),
    ("M5", "M5.3", run_test_m5_3),
    ("M5", "M5.4", run_test_m5_4),
    # M8
    ("M8", "M8.1", run_test_m8_1),
    ("M8", "M8.2", run_test_m8_2),
    # A6
    ("A6", "A6.1", run_test_a6_1),
    ("A6", "A6.2", run_test_a6_2),
    ("A6", "A6.3", run_test_a6_3),
]


def run_phase4_suite() -> int:
    start_time = time.time()
    print("=" * 80)
    print(" MedVoice v3.4 — Phase 4 (Φ-4) Master Verification Suite")
    print(" Tracks: M5, M8, A6 (9 subtasks | 45 rigorous test cases)")
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
    print(" Phase 4 Verification Results Summary")
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
        print("\n>>> ALL PHASE 4 SUBTASKS MET QUALITY GATE (45/45 PASSED). PROMOTION UNLOCKED. <<<\n")
        return 0
    else:
        print("\n>>> PHASE 4 QUALITY GATE FAILED. FIX DEFICIENCIES BEFORE PROCEEDING. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_phase4_suite())

