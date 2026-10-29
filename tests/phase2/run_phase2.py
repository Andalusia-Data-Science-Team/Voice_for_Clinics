"""
Phase 2 Master Test Orchestrator — Executes all Phase 2 Subtasks (B0, B1, B2, B3, B4, B5, M3).
Enforces:
  - 5 hard test cases per subtask (23 subtasks * 5 = 115 tests)
  - Results stored in test_results/{TASK}/{SUBTASK}/ ONLY after all 5 pass
  - Generates comprehensive Phase 2 execution summary report
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.phase2.test_b0 import run_test_b0_1, run_test_b0_2
from tests.phase2.test_b1 import run_test_b1_1, run_test_b1_2, run_test_b1_3, run_test_b1_4
from tests.phase2.test_b2 import run_test_b2_1, run_test_b2_2, run_test_b2_3
from tests.phase2.test_b3 import run_test_b3_1, run_test_b3_2, run_test_b3_3, run_test_b3_4
from tests.phase2.test_b4 import run_test_b4_1, run_test_b4_2, run_test_b4_3
from tests.phase2.test_b5 import run_test_b5_1, run_test_b5_2, run_test_b5_3
from tests.phase2.test_m3 import run_test_m3_1, run_test_m3_2, run_test_m3_3, run_test_m3_4, run_test_m3_5


def main():
    print("=" * 80)
    print("🚀 MEDVOICE v3.4 — PHASE 2 MASTER VERIFICATION ENGINE (CLINICAL CORE)")
    print("Enforcing: 5 Hard Tests / Subtask | Atomic Promotion on 5/5 Pass")
    print("=" * 80)

    subtask_runners = [
        # Track B0: Term Correction & Red-Flag Lexicon
        ("B0.1", run_test_b0_1),
        ("B0.2", run_test_b0_2),
        # Track B1: Structuring Prompt + Schema
        ("B1.1", run_test_b1_1),
        ("B1.2", run_test_b1_2),
        ("B1.3", run_test_b1_3),
        ("B1.4", run_test_b1_4),
        # Track B2: Extraction Prompts per Specialty
        ("B2.1", run_test_b2_1),
        ("B2.2", run_test_b2_2),
        ("B2.3", run_test_b2_3),
        # Track B3: Reasoning Prompt & ICD-10 Rules
        ("B3.1", run_test_b3_1),
        ("B3.2", run_test_b3_2),
        ("B3.3", run_test_b3_3),
        ("B3.4", run_test_b3_4),
        # Track B4: Reasoning Router
        ("B4.1", run_test_b4_1),
        ("B4.2", run_test_b4_2),
        ("B4.3", run_test_b4_3),
        # Track B5: Incremental Structuring
        ("B5.1", run_test_b5_1),
        ("B5.2", run_test_b5_2),
        ("B5.3", run_test_b5_3),
        # Track M3: Validation Layer
        ("M3.1", run_test_m3_1),
        ("M3.2", run_test_m3_2),
        ("M3.3", run_test_m3_3),
        ("M3.4", run_test_m3_4),
        ("M3.5", run_test_m3_5),
    ]

    total_subtasks = len(subtask_runners)
    passed_subtasks = 0
    total_cases = 0
    passed_cases = 0

    results = []

    t0_all = time.perf_counter()

    for subtask_id, runner_fn in subtask_runners:
        print(f"\n▶ Executing subtask [{subtask_id}]...")
        runner = runner_fn()
        sub_res = runner.finalize()
        results.append(sub_res)
        total_cases += len(sub_res.cases)
        passed_cases += sub_res.pass_count
        if sub_res.passed:
            passed_subtasks += 1

    total_time_ms = (time.perf_counter() - t0_all) * 1000.0

    print("\n" + "=" * 80)
    print("📊 PHASE 2 EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Subtasks:    {total_subtasks}")
    print(f"Verified (5/5):    {passed_subtasks} / {total_subtasks} ({(passed_subtasks/total_subtasks)*100:.1f}%)")
    print(f"Total Test Cases:  {passed_cases} / {total_cases} Passed ({(passed_cases/total_cases)*100:.1f}%)")
    print(f"Execution Time:    {total_time_ms:.1f} ms")
    print("=" * 80)

    # Print summary table
    print(f"{'Task':<6} {'Subtask':<8} {'Title':<45} {'Pass Rate':<12} {'Status'}")
    print("-" * 80)
    for r in results:
        status_str = "\033[92mVERIFIED\033[0m" if r.passed else "\033[91mFAILED\033[0m"
        print(f"{r.task_id:<6} {r.subtask_id:<8} {r.title[:42]:<45} {r.pass_count}/5 ({r.pass_count*20}%)   {status_str}")

    if passed_subtasks == total_subtasks:
        print("\n\033[92m🎉 PHASE 2 GATE CLEARED: All 23 Subtasks (115/115 Test Cases) Passed and Promoted to test_results/!\033[0m\n")
    else:
        print(f"\n\033[91m⚠️ PHASE 2 GATE BLOCKED: {total_subtasks - passed_subtasks} subtask(s) failed gate criteria.\033[0m\n")


if __name__ == "__main__":
    main()
