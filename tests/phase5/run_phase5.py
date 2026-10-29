"""
Phase 5 Master Orchestrator — Φ-5: Security, Benchmarks & Hardening
Runs all 10 subtasks (50 test cases) and prints a consolidated gate report.

Tracks:
  M9  — Penetration Test Fixes        (M9.1, M9.2)
  A8  — Gold Set Build                (A8.1)
  A9  — As-Is Baseline                (A9.1)
  B6  — Structuring Benchmarks        (B6.1, B6.2, B6.3)
  B8  — E2E New Pipeline on Gold Set  (B8.1)
  B9  — Fix Failed Cases & Re-run     (B9.1, B9.2)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Import all subtask runners ─────────────────────────────────────────────────
from tests.phase5.test_m9 import run_test_m9_1, run_test_m9_2
from tests.phase5.test_a8_a9 import run_test_a8_1, run_test_a9_1
from tests.phase5.test_b6_b8_b9 import (
    run_test_b6_1,
    run_test_b6_2,
    run_test_b6_3,
    run_test_b8_1,
    run_test_b9_1,
    run_test_b9_2,
)

PHASE_NAME = "Φ-5: Security, Benchmarks & Hardening"

SUBTASK_RUNNERS = [
    ("M9", "M9.1", "Penetration test — Fix high/medium findings",         run_test_m9_1),
    ("M9", "M9.2", "Penetration test — Retest with security team",         run_test_m9_2),
    ("A8", "A8.1", "Gold set build — Sample 100, de-identify, package",    run_test_a8_1),
    ("A9", "A9.1", "Baseline — Run v3.3 pipeline on gold set",             run_test_a9_1),
    ("B6", "B6.1", "Benchmark — Structuring fidelity metrics",             run_test_b6_1),
    ("B6", "B6.2", "Benchmark — Extraction model comparison",              run_test_b6_2),
    ("B6", "B6.3", "Benchmark — Reasoning escalation & memo",              run_test_b6_3),
    ("B8", "B8.1", "E2E pipeline on gold set (no crashes / schema fails)", run_test_b8_1),
    ("B9", "B9.1", "Fix failed cases: prompt, lexicon, routing adjustments",run_test_b9_1),
    ("B9", "B9.2", "Re-run affected encounters — zero regressions",        run_test_b9_2),
]

TOTAL_SUBTASKS = len(SUBTASK_RUNNERS)
TOTAL_CASES   = TOTAL_SUBTASKS * 5


def main() -> None:
    print(f"\n{'═' * 70}")
    print(f"  PHASE 5 MASTER ORCHESTRATOR — {PHASE_NAME}")
    print(f"  {TOTAL_SUBTASKS} subtasks × 5 test cases = {TOTAL_CASES} total")
    print(f"{'═' * 70}\n")

    results      = []
    cleared_count = 0
    failed_count  = 0
    start_ts      = time.perf_counter()

    for track, subtask_id, description, runner_fn in SUBTASK_RUNNERS:
        print(f"▶  [{track}] {subtask_id} — {description}")
        runner = runner_fn()
        res    = runner.finalize()
        results.append((track, subtask_id, description, res))
        if res.passed:
            cleared_count += 1
        else:
            failed_count += 1

    elapsed = time.perf_counter() - start_ts

    # ── Summary Table ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  {PHASE_NAME} — GATE REPORT")
    print(f"{'═' * 70}")
    print(f"  {'Track':<6} {'Subtask':<8} {'Gate':<10} {'Cases':<8} Description")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*35}")

    cases_passed = 0
    for track, subtask_id, description, res in results:
        gate   = "✅ CLEARED" if res.passed else "❌ REJECTED"
        n_pass = sum(1 for c in res.cases if c.passed)
        n_total = len(res.cases)
        cases_passed += n_pass
        print(f"  {track:<6} {subtask_id:<8} {gate:<10} {n_pass}/{n_total:<6} {description[:50]}")

    print(f"\n{'═' * 70}")
    print(f"  SUBTASKS CLEARED : {cleared_count}/{TOTAL_SUBTASKS}")
    print(f"  CASES PASSED     : {cases_passed}/{TOTAL_CASES}")
    print(f"  ELAPSED          : {elapsed:.2f}s")

    if cleared_count == TOTAL_SUBTASKS:
        print(f"\n  🏆  PHASE 5 GATE: ALL {TOTAL_SUBTASKS} SUBTASKS CLEARED — Φ-5 VERIFIED ✅")
    else:
        print(f"\n  ⛔  PHASE 5 GATE: {failed_count} SUBTASK(S) FAILED — NEEDS FIXES")
    print(f"{'═' * 70}\n")

    sys.exit(0 if cleared_count == TOTAL_SUBTASKS else 1)


if __name__ == "__main__":
    main()

