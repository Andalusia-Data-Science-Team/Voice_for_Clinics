"""
Phase 1 Master Test Orchestrator — Executes all Phase 1 Subtasks (A1, A2, A3, M1, M2).
Enforces:
  - 5 hard test cases per subtask (17 subtasks * 5 = 85 tests)
  - Results stored in test_results/{TASK}/{SUBTASK}/ ONLY after all 5 pass
  - Generates comprehensive execution summary report
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

from tests.phase1.test_a1 import run_test_a1_1, run_test_a1_2, run_test_a1_3, run_test_a1_4, run_test_a1_5
from tests.phase1.test_a2 import run_test_a2_1, run_test_a2_2, run_test_a2_3, run_test_a2_4
from tests.phase1.test_a3 import run_test_a3_1, run_test_a3_2, run_test_a3_3
from tests.phase1.test_m1 import run_test_m1_1, run_test_m1_2, run_test_m1_3
from tests.phase1.test_m2 import run_test_m2_1, run_test_m2_2


async def main():
    print("=" * 80)
    print("🚀 MEDVOICE v3.4 — PHASE 1 MASTER VERIFICATION ENGINE (DEV SUITE)")
    print("Enforcing: 5 Hard Tests / Subtask | Atomic Promotion on 5/5 Pass")
    print("=" * 80)

    subtask_runners = [
        # Track A1: LLM Concurrency
        ("A1.1", run_test_a1_1()),
        ("A1.2", run_test_a1_2()),
        ("A1.3", run_test_a1_3()),
        ("A1.4", run_test_a1_4()),
        ("A1.5", run_test_a1_5()),
        # Track A2: Azure Foundation
        ("A2.1", asyncio.to_thread(run_test_a2_1)),
        ("A2.2", asyncio.to_thread(run_test_a2_2)),
        ("A2.3", asyncio.to_thread(run_test_a2_3)),
        ("A2.4", asyncio.to_thread(run_test_a2_4)),
        # Track A3: Deploy Foundry Models
        ("A3.1", run_test_a3_1()),
        ("A3.2", asyncio.to_thread(run_test_a3_2)),
        ("A3.3", asyncio.to_thread(run_test_a3_3)),
        # Track M1: Clinical AI Gateway
        ("M1.1", asyncio.to_thread(run_test_m1_1)),
        ("M1.2", asyncio.to_thread(run_test_m1_2)),
        ("M1.3", asyncio.to_thread(run_test_m1_3)),
        # Track M2: Provider Adapter
        ("M2.1", asyncio.to_thread(run_test_m2_1)),
        ("M2.2", asyncio.to_thread(run_test_m2_2)),
    ]

    total_subtasks = len(subtask_runners)
    passed_subtasks = 0
    total_cases = 0
    passed_cases = 0

    results = []

    t0_all = time.perf_counter()

    for subtask_id, runner_item in subtask_runners:
        print(f"\n▶ Executing subtask [{subtask_id}]...")
        if asyncio.iscoroutine(runner_item):
            runner = await runner_item
        elif callable(runner_item):
            runner = await runner_item()
        else:
            runner = runner_item

        sub_res = runner.finalize()
        results.append(sub_res)
        total_cases += len(sub_res.cases)
        passed_cases += sub_res.pass_count
        if sub_res.passed:
            passed_subtasks += 1

    total_time_ms = (time.perf_counter() - t0_all) * 1000.0

    print("\n" + "=" * 80)
    print("📊 PHASE 1 EXECUTION SUMMARY")
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
        print("\n\033[92m🎉 PHASE 1 GATE CLEARED: All 17 Subtasks (85/85 Test Cases) Passed and Promoted to test_results/!\033[0m\n")
    else:
        print(f"\n\033[91m⚠️ PHASE 1 GATE BLOCKED: {total_subtasks - passed_subtasks} subtask(s) failed gate criteria.\033[0m\n")


if __name__ == "__main__":
    asyncio.run(main())

