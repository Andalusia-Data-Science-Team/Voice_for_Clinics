"""
MedVoice Phase 1 Test Execution Harness & Gatekeeper.

Enforces:
1. Every subtask has exactly 5 distinct hard test cases.
2. Results are executed and staged in `.staging_test_results/{TASK}/{SUBTASK}/`.
3. ATOMIC PROMOTION: Only if ALL 5 test cases pass (5/5) are results promoted to
   `test_results/{TASK}/{SUBTASK}/`.
4. Saves full test script, structured `test_report.json`, styled `test_report.html`,
   and execution logs.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
STAGING_DIR = ROOT_DIR / ".staging_test_results"
RESULTS_DIR = ROOT_DIR / "test_results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclasses.dataclass
class TestCaseResult:
    case_num: int
    name: str
    input_setup: str
    expected_result: str
    severity: str
    passed: bool
    duration_ms: float
    evidence: str = ""
    error: Optional[str] = None


@dataclasses.dataclass
class SubtaskResult:
    task_id: str
    subtask_id: str
    title: str
    target_files: List[str]
    cases: List[TestCaseResult]
    start_time: str
    end_time: str
    total_duration_ms: float

    @property
    def passed(self) -> bool:
        return len(self.cases) == 5 and all(c.passed for c in self.cases)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)


def render_html_report(res: SubtaskResult) -> str:
    status_color = "#10b981" if res.passed else "#ef4444"
    status_label = "ALL 5 PASSED — VERIFIED" if res.passed else f"FAILED ({res.pass_count}/5 PASSED)"
    rows = []
    for c in res.cases:
        c_badge = "<span style='background:#10b981;color:#fff;padding:3px 8px;border-radius:4px;font-weight:600;'>PASS</span>" if c.passed else "<span style='background:#ef4444;color:#fff;padding:3px 8px;border-radius:4px;font-weight:600;'>FAIL</span>"
        sev_color = "#dc2626" if c.severity == "Crit" else "#ea580c" if c.severity == "High" else "#2563eb"
        sev_badge = f"<span style='background:{sev_color};color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:700;'>{c.severity}</span>"
        err_block = f"<div style='margin-top:6px;background:#fef2f2;border-left:3px solid #ef4444;padding:8px;font-family:monospace;font-size:12px;color:#991b1b;white-space:pre-wrap;'>{c.error}</div>" if c.error else ""
        evidence_block = f"<div style='margin-top:4px;font-size:12px;color:#4b5563;font-family:monospace;'>{c.evidence}</div>" if c.evidence else ""
        rows.append(f"""
        <tr style="border-bottom:1px solid #e5e7eb;">
            <td style="padding:12px;font-weight:600;text-align:center;">#{c.case_num}</td>
            <td style="padding:12px;text-align:center;">{sev_badge}</td>
            <td style="padding:12px;"><strong>{c.name}</strong></td>
            <td style="padding:12px;color:#374151;font-size:13px;">{c.input_setup}</td>
            <td style="padding:12px;color:#374151;font-size:13px;">{c.expected_result}</td>
            <td style="padding:12px;text-align:center;">{c_badge}</td>
            <td style="padding:12px;text-align:right;font-family:monospace;">{c.duration_ms:.1f}ms</td>
        </tr>
        <tr>
            <td colspan="7" style="padding:0 12px 12px 12px;background:#fafafa;">
                {evidence_block}
                {err_block}
            </td>
        </tr>
        """)

    files_badges = " ".join([f"<code style='background:#e5e7eb;padding:2px 6px;border-radius:3px;font-size:12px;'>{f}</code>" for f in res.target_files])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Report: {res.task_id} - {res.subtask_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
        .container {{ max-width: 1080px; margin: 0 auto; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); padding: 32px; border: 1px solid #e2e8f0; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 20px; }}
        .badge-status {{ background: {status_color}; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; padding: 16px; background: #f1f5f9; border-radius: 6px; }}
        .meta-card {{ display: flex; flex-direction: column; }}
        .meta-label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; }}
        .meta-value {{ font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #f8fafc; padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; color: #64748b; border-bottom: 2px solid #cbd5e1; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin:0 0 8px 0; font-size: 24px;">Subtask Verification: {res.subtask_id}</h1>
                <div style="font-size: 15px; color: #475569;">{res.title}</div>
            </div>
            <div class="badge-status">{status_label}</div>
        </div>

        <div class="meta-grid">
            <div class="meta-card">
                <span class="meta-label">Task Identifier</span>
                <span class="meta-value">{res.task_id} / {res.subtask_id}</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">Pass Rate</span>
                <span class="meta-value">{res.pass_count} / {len(res.cases)} (100% required)</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">Execution Time</span>
                <span class="meta-value">{res.total_duration_ms:.1f} ms</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">Executed At</span>
                <span class="meta-value">{res.end_time}</span>
            </div>
        </div>

        <div style="margin: 16px 0; font-size: 13px;">
            <strong>Target Files:</strong> {files_badges}
        </div>

        <table>
            <thead>
                <tr>
                    <th style="text-align:center;width:40px;">#</th>
                    <th style="text-align:center;width:60px;">Sev</th>
                    <th>Test Case</th>
                    <th>Input / Setup</th>
                    <th>Expected Result</th>
                    <th style="text-align:center;width:80px;">Status</th>
                    <th style="text-align:right;width:80px;">Time</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
</body>
</html>"""


class SubtaskRunner:
    def __init__(self, task_id: str, subtask_id: str, title: str, target_files: List[str], source_file: str):
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.title = title
        self.target_files = target_files
        self.source_file = source_file
        self.cases: List[TestCaseResult] = []

    def record_case(
        self,
        case_num: int,
        name: str,
        input_setup: str,
        expected_result: str,
        severity: str,
        fn: Callable[[], Any],
    ) -> bool:
        t0 = time.perf_counter()
        passed = False
        evidence = ""
        err_msg = None
        try:
            res = fn()
            if isinstance(res, tuple):
                passed, evidence = res[0], str(res[1])
            elif res is False:
                passed = False
                evidence = "Returned False"
            else:
                passed = True
                evidence = str(res) if res is not None else "Passed assertion"
        except Exception as e:
            passed = False
            err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            dur = (time.perf_counter() - t0) * 1000.0
            tc = TestCaseResult(
                case_num=case_num,
                name=name,
                input_setup=input_setup,
                expected_result=expected_result,
                severity=severity,
                passed=passed,
                duration_ms=dur,
                evidence=evidence,
                error=err_msg,
            )
            self.cases.append(tc)
            return passed

    async def record_case_async(
        self,
        case_num: int,
        name: str,
        input_setup: str,
        expected_result: str,
        severity: str,
        coro_fn: Callable[[], Any],
    ) -> bool:
        t0 = time.perf_counter()
        passed = False
        evidence = ""
        err_msg = None
        try:
            res = await coro_fn()
            if isinstance(res, tuple):
                passed, evidence = res[0], str(res[1])
            elif res is False:
                passed = False
                evidence = "Returned False"
            else:
                passed = True
                evidence = str(res) if res is not None else "Passed assertion"
        except Exception as e:
            passed = False
            err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            dur = (time.perf_counter() - t0) * 1000.0
            tc = TestCaseResult(
                case_num=case_num,
                name=name,
                input_setup=input_setup,
                expected_result=expected_result,
                severity=severity,
                passed=passed,
                duration_ms=dur,
                evidence=evidence,
                error=err_msg,
            )
            self.cases.append(tc)
            return passed

    def finalize(self) -> SubtaskResult:
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        total_dur = sum(c.duration_ms for c in self.cases)
        sub_res = SubtaskResult(
            task_id=self.task_id,
            subtask_id=self.subtask_id,
            title=self.title,
            target_files=self.target_files,
            cases=self.cases,
            start_time=now_str,
            end_time=now_str,
            total_duration_ms=total_dur,
        )

        staging_subtask_dir = STAGING_DIR / self.task_id / self.subtask_id
        final_subtask_dir = RESULTS_DIR / self.task_id / self.subtask_id

        # Always wipe staging
        if staging_subtask_dir.exists():
            shutil.rmtree(staging_subtask_dir)
        staging_subtask_dir.mkdir(parents=True, exist_ok=True)

        report_dict = {
            "task_id": sub_res.task_id,
            "subtask_id": sub_res.subtask_id,
            "title": sub_res.title,
            "target_files": sub_res.target_files,
            "passed": sub_res.passed,
            "pass_count": sub_res.pass_count,
            "total_cases": len(sub_res.cases),
            "total_duration_ms": sub_res.total_duration_ms,
            "timestamp": sub_res.end_time,
            "test_cases": [dataclasses.asdict(c) for c in sub_res.cases],
        }

        # Write to staging
        (staging_subtask_dir / "test_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        (staging_subtask_dir / "test_report.html").write_text(render_html_report(sub_res), encoding="utf-8")
        if os.path.exists(self.source_file):
            shutil.copy2(self.source_file, staging_subtask_dir / "test_suite.py")

        # GATE RULE: Only promote to test_results/ if ALL 5 PASSED
        if sub_res.passed:
            if final_subtask_dir.exists():
                shutil.rmtree(final_subtask_dir)
            final_subtask_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging_subtask_dir, final_subtask_dir)
            print(f"\033[92m[GATE CLEARED]\033[0m {self.subtask_id}: 5/5 PASSED -> Stored in {final_subtask_dir}")
        else:
            print(f"\033[91m[GATE REJECTED]\033[0m {self.subtask_id}: {sub_res.pass_count}/5 PASSED -> NOT stored in test_results/")
            for c in sub_res.cases:
                if not c.passed:
                    print(f"  \033[91mFAILED TC{c.case_num}: {c.name}\033[0m: {c.error or c.evidence}")

        return sub_res
