"""Evaluate extraction and diagnosis models against the six real-use cases.

This is an opt-in live evaluation suite. It sends the workbook scenarios to
all configured extraction and reasoning models, records latency and provider
reported cost, and writes a workbook with one row per scenario/model/field.
No LLM judge is required: the report preserves the raw model outputs so they
can be compared side by side.

Usage::

    RUN_REAL_USE_CASE_EVAL=1 \
        .venv/bin/python -m pytest tests/test_real_use_case_models.py -s

The report path is controlled by ``REAL_USE_CASE_REPORT`` and defaults to
``reports/real_use_case_model_evaluation.xlsx``. Set
``REAL_USE_CASE_FAIL_ON_ERROR=1`` to fail the test when any model call fails;
otherwise failures are included in the report and the test remains usable for
partial comparisons.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl
import pytest
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from core.model_registry import EXTRACTION_MODELS, REASONING_MODELS, ModelSpec
from llm.runtime import LLMRuntime
from model.llm_service import LLMService

WORKBOOK_PATH = Path(os.environ.get(
    "REAL_USE_CASE_XLSX",
    "/home/ai/Workspace/Doha/MedVoice/src/data/real_use_cases.xlsx",
))
REPORT_PATH = Path(os.environ.get(
    "REAL_USE_CASE_REPORT",
    "reports/real_use_case_model_evaluation.xlsx",
))
RUN_EVALUATION = os.environ.get("RUN_REAL_USE_CASE_EVAL", "").lower() in {
    "1", "true", "yes",
}
FAIL_ON_ERROR = os.environ.get("REAL_USE_CASE_FAIL_ON_ERROR", "").lower() in {
    "1", "true", "yes",
}

SCENARIO_SHEETS = {
    "er": "ER Scenarios",
    "opd": "OPD Scenarios",
    "dental": "Dental Scenarios",
}


@dataclass(frozen=True)
class EvaluationScenario:
    """One workbook scenario and the transcript sent to the models."""

    scenario_id: str
    department: str
    language: str
    transcript: str
    expected_fields: dict[str, str]


@dataclass
class EvaluationRow:
    """One output cell or diagnosis result in the generated report."""

    scenario_id: str
    department: str
    language: str
    stage: str
    model_id: str
    model_label: str
    model_slug: str
    field: str
    expected: str
    output: str
    latency_sec: float | None
    cost_usd: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str = ""


def _normalise(value: Any) -> str:
    """Convert workbook values to trimmed strings."""
    return "" if value is None else str(value).replace("\xa0", " ").strip()


def _field_key(label: str) -> str:
    """Convert a workbook field label to the pipeline's JSON naming style."""
    aliases = {
        "chief complaint (c/o)": "chief_complaint",
        "comorbidities": "comorbidities",
        "icd-10 codes": "icd10_codes",
        "final diagnosis": "final_diagnosis",
        "triage classification": "triage_priority",
        "triage priority": "triage_priority",
        "consent obtained": "consent_obtained",
        "radiographs taken": "radiographs_taken",
        "tooth number": "tooth_number",
    }
    normalised = " ".join(label.lower().replace("\xa0", " ").split())
    if normalised in aliases:
        return aliases[normalised]
    return normalised.replace("/", " ").replace("-", " ").replace("(", "").replace(")", "").replace(" ", "_")


def _scenario_id(value: str) -> str:
    """Normalize workbook scenario IDs such as ``Dental-1`` to ``DENTAL-1``."""
    return value.strip().upper()


def _parse_scripts(workbook: openpyxl.Workbook) -> dict[str, dict[str, str]]:
    """Parse the Scripts sheet and assign ER/OPD/Dental scenario IDs."""
    rows = list(workbook["Scripts"].iter_rows(values_only=True))
    header = {_normalise(value): index for index, value in enumerate(rows[0])}
    counts: dict[str, int] = {}
    scripts: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        department = _normalise(row[header["Departement"]]).lower()
        transcript = _normalise(row[header["Scenario"]])
        if not department or not transcript:
            continue
        counts[department] = counts.get(department, 0) + 1
        scenario_id = _scenario_id(f"{department}-{counts[department]}")
        scripts[scenario_id] = {
            "department": department,
            "language": "ar" if _normalise(row[header["Language"]]).lower().startswith("ar") else "en",
            "transcript": transcript,
        }
    return scripts


def _parse_expected_sheet(worksheet: openpyxl.worksheet.worksheet.Worksheet) -> dict[str, dict[str, str]]:
    """Parse scenario field/expected tables from one workbook sheet."""
    scenarios: dict[str, dict[str, str]] = {}
    current_id = ""
    for row in worksheet.iter_rows(values_only=True):
        first = _normalise(row[0] if row else None)
        second = _normalise(row[1] if len(row) > 1 else None)
        if first and not second and "-" in first:
            current_id = _scenario_id(first)
            scenarios.setdefault(current_id, {})
        elif current_id and first.lower() != "field" and second:
            scenarios[current_id][_field_key(first)] = second
    return scenarios


def load_scenarios(path: Path = WORKBOOK_PATH) -> list[EvaluationScenario]:
    """Load the six workbook scenarios in script order."""
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    scripts = _parse_scripts(workbook)
    expected: dict[str, dict[str, str]] = {}
    for worksheet_name in SCENARIO_SHEETS.values():
        expected.update(_parse_expected_sheet(workbook[worksheet_name]))
    return [
        EvaluationScenario(
            scenario_id=scenario_id,
            department=meta["department"],
            language=meta["language"],
            transcript=meta["transcript"],
            expected_fields=expected.get(scenario_id, {}),
        )
        for scenario_id, meta in scripts.items()
    ]


def _usage_value(meta: dict[str, Any] | None, key: str) -> Any:
    """Read one optional provider usage value."""
    return meta.get(key) if meta else None


def _text(value: Any) -> str:
    """Serialize model output safely for an Excel cell."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _rows_for_extraction(
    scenario: EvaluationScenario,
    model: ModelSpec,
    output: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    error: str = "",
) -> list[EvaluationRow]:
    """Create one report row per expected/extracted clinical field."""
    data = (output or {}).get("json_data", {}) if isinstance(output, dict) else {}
    fields = sorted(set(scenario.expected_fields) | set(data))
    return [
        EvaluationRow(
            scenario_id=scenario.scenario_id,
            department=scenario.department,
            language=scenario.language,
            stage="Extraction",
            model_id=model.id,
            model_label=model.label,
            model_slug=model.slug,
            field=field,
            expected=scenario.expected_fields.get(field, ""),
            output=_text(data.get(field)),
            latency_sec=_usage_value(meta, "latency_sec"),
            cost_usd=_usage_value(meta, "cost_usd"),
            prompt_tokens=_usage_value(meta, "prompt_tokens"),
            completion_tokens=_usage_value(meta, "completion_tokens"),
            error=error,
        )
        for field in fields
    ]


def _rows_for_reasoning(
    scenario: EvaluationScenario,
    model: ModelSpec,
    output: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    error: str = "",
) -> list[EvaluationRow]:
    """Create report rows for ICD-10 recommendations and diagnosis reasoning."""
    data = (output or {}).get("json_data", {}) if isinstance(output, dict) else {}
    expected_fields = {
        key: value for key, value in scenario.expected_fields.items()
        if key in {"icd10_codes", "final_diagnosis", "reasoning", "triage_priority", "differential_diagnoses"}
    }
    fields = sorted(set(expected_fields) | {"icd10_codes", "final_diagnosis", "reasoning"})
    return [
        EvaluationRow(
            scenario_id=scenario.scenario_id,
            department=scenario.department,
            language=scenario.language,
            stage="Reasoning",
            model_id=model.id,
            model_label=model.label,
            model_slug=model.slug,
            field=field,
            expected=expected_fields.get(field, ""),
            output=_text(data.get(field)),
            latency_sec=_usage_value(meta, "latency_sec"),
            cost_usd=_usage_value(meta, "cost_usd"),
            prompt_tokens=_usage_value(meta, "prompt_tokens"),
            completion_tokens=_usage_value(meta, "completion_tokens"),
            error=error,
        )
        for field in fields
    ]


def _write_report(rows: list[EvaluationRow], path: Path) -> None:
    """Write detailed, summary, and error sheets to an XLSX report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    details = workbook.active
    details.title = "Detailed Results"
    headers = [
        "Scenario", "Department", "Language", "Stage", "Model ID", "Model",
        "Model Slug", "Field", "Expected", "Output", "Latency (sec)",
        "Cost (USD)", "Prompt Tokens", "Completion Tokens", "Error",
    ]
    details.append(headers)
    for row in rows:
        details.append([
            row.scenario_id, row.department, row.language, row.stage, row.model_id,
            row.model_label, row.model_slug, row.field, row.expected, row.output,
            row.latency_sec, row.cost_usd, row.prompt_tokens, row.completion_tokens,
            row.error,
        ])

    summary = workbook.create_sheet("Summary")
    summary.append(["Scenario", "Department", "Stage", "Model", "Model Slug", "Latency (sec)", "Cost (USD)", "Errors"])
    grouped: dict[tuple[str, str, str, str, str], list[EvaluationRow]] = {}
    for row in rows:
        key = (row.scenario_id, row.department, row.stage, row.model_label, row.model_slug)
        grouped.setdefault(key, []).append(row)
    for key, values in grouped.items():
        latencies = [v.latency_sec for v in values if v.latency_sec is not None]
        costs = [v.cost_usd for v in values if v.cost_usd is not None]
        errors = sorted({v.error for v in values if v.error})
        summary.append([*key, max(latencies, default=None), max(costs, default=None), " | ".join(errors)])

    errors_sheet = workbook.create_sheet("Errors")
    errors_sheet.append(headers)
    for row in rows:
        if row.error:
            errors_sheet.append([
                row.scenario_id, row.department, row.language, row.stage, row.model_id,
                row.model_label, row.model_slug, row.field, row.expected, row.output,
                row.latency_sec, row.cost_usd, row.prompt_tokens, row.completion_tokens,
                row.error,
            ])

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="6B4F35")
            cell.alignment = Alignment(horizontal="center")
        for column in worksheet.columns:
            letter = get_column_letter(column[0].column)
            worksheet.column_dimensions[letter].width = min(
                55, max(12, max(len(str(cell.value or "")) for cell in column) + 2),
            )
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

    workbook.save(path)


@pytest.mark.llm
@pytest.mark.asyncio
@pytest.mark.skipif(not RUN_EVALUATION, reason="Set RUN_REAL_USE_CASE_EVAL=1 to run live model evaluation")
async def test_real_use_case_models_generate_report():
    """Evaluate all configured extraction/reasoning models and write XLSX output."""
    if not WORKBOOK_PATH.exists():
        pytest.fail(f"Workbook not found: {WORKBOOK_PATH}")

    scenarios = load_scenarios()
    assert len(scenarios) == 6, f"Expected six scenarios, found {len(scenarios)}"
    runtime = LLMRuntime()
    rows: list[EvaluationRow] = []
    try:
        for scenario in scenarios:
            for model in EXTRACTION_MODELS.values():
                meta: dict[str, Any] = {}
                started = time.perf_counter()
                try:
                    output = await LLMService.extract_features_async(
                        translated_text=scenario.transcript,
                        runtime=runtime,
                        department=scenario.department,
                        model_id=model.id,
                        meta=meta,
                    )
                    meta.setdefault(
                        "latency_sec", round(time.perf_counter() - started, 3)
                    )
                    rows.extend(_rows_for_extraction(scenario, model, output, meta))
                except Exception as exc:
                    rows.extend(_rows_for_extraction(scenario, model, None, meta, str(exc)))

            for model in REASONING_MODELS.values():
                meta = {}
                started = time.perf_counter()
                try:
                    output = await LLMService.recommend_diagnosis_async(
                        translated_text=scenario.transcript,
                        runtime=runtime,
                        department=scenario.department,
                        model_id=model.id,
                        meta=meta,
                    )
                    meta.setdefault(
                        "latency_sec", round(time.perf_counter() - started, 3)
                    )
                    rows.extend(_rows_for_reasoning(scenario, model, output, meta))
                except Exception as exc:
                    rows.extend(_rows_for_reasoning(scenario, model, None, meta, str(exc)))
    finally:
        await runtime.close()

    _write_report(rows, REPORT_PATH)
    print(f"Wrote {len(rows)} result rows to {REPORT_PATH}")
    errors = sorted({row.error for row in rows if row.error})
    if FAIL_ON_ERROR and errors:
        pytest.fail("Model evaluation errors:\n" + "\n".join(errors))
