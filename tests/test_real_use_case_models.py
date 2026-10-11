"""Evaluate extraction and diagnosis models against the six real-use cases.

This is an opt-in live evaluation suite. It sends the workbook scenarios to
all configured extraction and reasoning models, records latency and provider
reported cost, and writes a workbook with one row per scenario/model/field.
Each output is also graded against the expected value by a cross-family judge:
Claude Opus 5.5 judges OpenAI candidates, and GPT 5.6 Sol judges all other
candidates. The report contains one score and one justification per output.

Usage::

    RUN_REAL_USE_CASE_EVAL=1 \
        .venv/bin/python -m pytest tests/test_real_use_case_models.py -s

The report path is controlled by ``REAL_USE_CASE_REPORT`` and defaults to
``reports/real_use_case_model_evaluation.xlsx``. Set
``REAL_USE_CASE_FAIL_ON_ERROR=1`` to fail the test when any model call fails;
otherwise failures are included in the report and the test remains usable for
partial comparisons. Judge slugs can be overridden with
``JUDGE_SLUG_FOR_OPENAI`` and ``JUDGE_SLUG_FOR_NON_OPENAI``.
Set ``REAL_USE_CASE_SCENARIO=ER-1`` to evaluate only one scenario first.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl
import pytest
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from core.model_registry import (
    EXTRACTION_MODELS,
    REASONING_MODELS,
    ModelSpec,
    judge_for,
)
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
SCENARIO_FILTER = os.environ.get("REAL_USE_CASE_SCENARIO", "").strip().upper()

SCENARIO_SHEETS = {
    "er": "ER Scenarios",
    "opd": "OPD Scenarios",
    "dental": "Dental Scenarios",
}

# Workbook labels are normalized to the pipeline's canonical JSON keys. The
# display label is kept separately so synonymous keys produce one readable
# row, for example ``plan_of_care`` and
# ``plan_of_care_short_long_term`` become ``Plan of Care``.
FIELD_ALIASES = {
    "plan_of_care": "plan_of_care",
    "plan_of_care_short_long_term": "plan_of_care",
}
FIELD_DISPLAY_NAMES = {
    "plan_of_care": "Plan of Care",
    "icd10_codes": "ICD-10 Codes",
    "final_diagnosis": "Final Diagnosis",
    "triage_priority": "Triage Priority",
    "differential_diagnoses": "Differential Diagnoses",
}

EMPTY_VALUE_RE = re.compile(
    r"^(?:none|blank|n/?a|empty|nil|not specified|not reported|"
    r"not mentioned|not noted|not available|not documented|unknown|"
    r"unavailable|no information|no data)$",
    re.IGNORECASE,
)
EMPTY_PHRASE_RE = re.compile(
    r"^(?:not\s+(?:specified|applicable|mentioned|reported|noted|"
    r"available|documented)|no\s+(?:information|data|finding|findings|"
    r"abnormality|abnormalities)|no\b.*\b(?:reported|noted|specified|"
    r"found|identified|documented|observed|detected|mentioned))$",
    re.IGNORECASE,
)


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
    score: float | None = None
    justification: str = ""
    judge_model: str = ""
    field_in_workbook: bool = True
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
        key = aliases[normalised]
    else:
        key = normalised.replace("/", " ").replace("-", " ")
        key = key.replace("(", "").replace(")", "").replace(" ", "_")
    return FIELD_ALIASES.get(key, key)


def _display_field_name(key: str) -> str:
    """Return one human-readable field label for the report."""
    if key in FIELD_DISPLAY_NAMES:
        return FIELD_DISPLAY_NAMES[key]
    return key.replace("_", " ").title()


def _is_blank_like(value: str) -> bool:
    """Return whether a value means that no information was provided."""
    text = str(value or "").strip().rstrip(".").strip()
    if not text:
        return True
    text = re.sub(r"\([^)]*\)", "", text).strip()
    return bool(EMPTY_VALUE_RE.fullmatch(text) or EMPTY_PHRASE_RE.fullmatch(text))


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
            "language": (
                "ar"
                if _normalise(row[header["Language"]]).lower().startswith("ar")
                else "en"
            ),
            "transcript": transcript,
        }
    return scripts


def _parse_expected_sheet(
    worksheet: openpyxl.worksheet.worksheet.Worksheet,
) -> dict[str, dict[str, str]]:
    """Parse scenario field/expected tables from one workbook sheet."""
    scenarios: dict[str, dict[str, str]] = {}
    current_id = ""
    for row in worksheet.iter_rows(values_only=True):
        first = _normalise(row[0] if row else None)
        second_raw = row[1] if len(row) > 1 else None
        second = _normalise(second_raw)
        if first and second_raw is None and "-" in first:
            current_id = _scenario_id(first)
            scenarios.setdefault(current_id, {})
        elif current_id and first.lower() != "field" and second_raw is not None:
            scenarios[current_id][_field_key(first)] = second
    return scenarios


def load_scenarios(path: Path = WORKBOOK_PATH) -> list[EvaluationScenario]:
    """Load the six workbook scenarios in script order."""
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    scripts = _parse_scripts(workbook)
    expected: dict[str, dict[str, str]] = {}
    for worksheet_name in SCENARIO_SHEETS.values():
        for scenario_id, fields in _parse_expected_sheet(
            workbook[worksheet_name]
        ).items():
            target = expected.setdefault(scenario_id, {})
            for key, value in fields.items():
                # If two synonymous workbook labels occur, retain one
                # unified row and combine the reference text when needed.
                if key in target and target[key] and value and target[key] != value:
                    target[key] = f"{target[key]} / {value}"
                elif key not in target or value:
                    target[key] = value
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


def _judge_prompt(
    scenario_id: str,
    stage: str,
    items: list[EvaluationRow],
) -> str:
    """Build a strict prompt for scoring one model result against expected text."""
    fields = "\n\n".join(
        f"field_id: {row.field!r}\n"
        f"expected: {row.expected!r}\nactual: {row.output!r}"
        for row in items
    )
    return (
        f"You are evaluating a medical AI {stage.lower()} result for scenario "
        f"{scenario_id}.\n\n"
        "Compare each model output with its expected reference.\n"
        "Score clinical correctness and completeness, not wording, length, or "
        "formatting.\n"
        "Equivalent medical paraphrases should receive the same score as the "
        "reference.\n"
        "Penalize missing, contradicted, or unsupported clinical facts.\n\n"
        f"Fields:\n{fields}\n\n"
        "Return ONLY a JSON object with:\n"
        "{\n"
        '  "field_id": {\n'
        '    "score": integer from 0 to 100,\n'
        '    "justification": "brief explanation of the score"\n'
        "  }\n"
        "}"
    )


def _message_content(response: Any) -> str:
    """Extract text from a chat response, including multipart content."""
    if not response.choices:
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts).strip()
    return str(content or "").strip()


def _parse_judge_json(content: str) -> dict[str, Any]:
    """Parse JSON returned as plain text or inside a Markdown code fence."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge response JSON was not an object")
    return parsed


def _judge_empty_response_reason(response: Any) -> str:
    """Describe an empty provider response without logging prompt content."""
    finish = (
        getattr(response.choices[0], "finish_reason", None)
        if response.choices
        else None
    )
    message = response.choices[0].message if response.choices else None
    refusal = getattr(message, "refusal", None)
    return f"empty judge response (finish_reason={finish!r}, refusal={refusal!r})"


async def _judge_output(
    runtime: LLMRuntime,
    judge: ModelSpec,
    scenario_id: str,
    stage: str,
    rows: list[EvaluationRow],
) -> dict[str, tuple[float | None, str]]:
    """Score all fields from one candidate result with one judge call."""
    prompt = _judge_prompt(scenario_id, stage, rows)
    request = {
        "model": judge.slug,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1000, 250 * len(rows)),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    response = await runtime.complete(
        model=judge.slug,
        create_kwargs=request,
    )
    content = _message_content(response)
    # Some OpenRouter routes for Claude return an empty content field when the
    # JSON response-format hint is not supported. Retry the same judge request
    # without that optional hint; the prompt still requires JSON.
    if not content:
        request.pop("response_format", None)
        response = await runtime.complete(model=judge.slug, create_kwargs=request)
        content = _message_content(response)
    try:
        if not content:
            raise ValueError(_judge_empty_response_reason(response))
        result = _parse_judge_json(content)
        scored: dict[str, tuple[float | None, str]] = {}
        for row in rows:
            item = result.get(row.field, {})
            try:
                score = max(0.0, min(100.0, float(item["score"])))
                justification = str(item.get("justification", ""))
            except (KeyError, TypeError, ValueError):
                score = None
                justification = "Judge omitted a valid score for this field."
            scored[row.field] = (score, justification)
        return scored
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        reason = f"Judge response could not be parsed: {exc}"
        return {row.field: (None, reason) for row in rows}


async def _score_rows(
    runtime: LLMRuntime,
    rows: list[EvaluationRow],
) -> None:
    """Fill score and justification for every model output row."""
    candidates = {
        (model.id, model.slug): model
        for model in (*EXTRACTION_MODELS.values(), *REASONING_MODELS.values())
    }
    groups: dict[tuple[str, str, str, str], list[EvaluationRow]] = {}
    for row in rows:
        groups.setdefault(
            (row.scenario_id, row.stage, row.model_id, row.model_slug), []
        ).append(row)

    for (scenario_id, stage, model_id, model_slug), group in groups.items():
        candidate = candidates.get((model_id, model_slug))
        if candidate is None:
            for row in group:
                row.justification = "Candidate model was not found in the registry."
            continue
        judge = judge_for(candidate)
        for row in group:
            row.judge_model = judge.label
            if not row.field_in_workbook:
                row.justification = "Not scored: field is absent from the workbook."
                continue
            if row.error:
                row.justification = (
                    "Not scored because the candidate model call failed."
                )
                continue
            if _is_blank_like(row.expected):
                row.score = 100.0 if _is_blank_like(row.output) else 20.0
                row.justification = (
                    "Blank-equivalent value as expected."
                    if row.score == 100.0
                    else "Expected a blank-equivalent value, but model populated it."
                )
                continue
            if _is_blank_like(row.output):
                row.score = 0.0
                row.justification = "Missing expected content."
                continue
        try:
            judgeable = [
                row for row in group
                if (
                    not row.error
                    and row.field_in_workbook
                    and not _is_blank_like(row.expected)
                    and not _is_blank_like(row.output)
                )
            ]
            if not judgeable:
                continue
            scored = await _judge_output(
                runtime, judge, scenario_id, stage, judgeable
            )
            for row in judgeable:
                row.score, row.justification = scored.get(
                    row.field, (None, "Judge omitted this field.")
                )
        except Exception as exc:
            for row in group:
                if row.error:
                    continue
                row.justification = f"Judge call failed: {exc}"


def _rows_for_extraction(
    scenario: EvaluationScenario,
    model: ModelSpec,
    output: dict[str, Any] | None,
    meta: dict[str, Any] | None,
    error: str = "",
) -> list[EvaluationRow]:
    """Create one report row per expected/extracted clinical field."""
    data = (output or {}).get("json_data", {}) if isinstance(output, dict) else {}
    # Include model-only keys for visibility, but leave their expected value,
    # score, and justification empty because the workbook does not define them.
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
            field=_display_field_name(field),
            expected=scenario.expected_fields.get(field, ""),
            output=_text(data.get(field)),
            latency_sec=_usage_value(meta, "latency_sec"),
            cost_usd=_usage_value(meta, "cost_usd"),
            prompt_tokens=_usage_value(meta, "prompt_tokens"),
            completion_tokens=_usage_value(meta, "completion_tokens"),
            field_in_workbook=field in scenario.expected_fields,
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
    reasoning_fields = {
        "icd10_codes",
        "final_diagnosis",
        "reasoning",
        "triage_priority",
        "differential_diagnoses",
    }
    expected_fields = {
        key: value for key, value in scenario.expected_fields.items()
        if key in reasoning_fields
    }
    # Include additional model output fields for visibility without scoring
    # them when they are absent from the workbook.
    fields = sorted(set(expected_fields) | set(data))
    return [
        EvaluationRow(
            scenario_id=scenario.scenario_id,
            department=scenario.department,
            language=scenario.language,
            stage="Reasoning",
            model_id=model.id,
            model_label=model.label,
            model_slug=model.slug,
            field=_display_field_name(field),
            expected=scenario.expected_fields.get(field, ""),
            output=_text(data.get(field)),
            latency_sec=_usage_value(meta, "latency_sec"),
            cost_usd=_usage_value(meta, "cost_usd"),
            prompt_tokens=_usage_value(meta, "prompt_tokens"),
            completion_tokens=_usage_value(meta, "completion_tokens"),
            field_in_workbook=field in expected_fields,
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
        "Model Slug", "Field", "Expected", "Output", "Score (0-100)",
        "Justification", "Judge Model", "Latency (sec)", "Cost (USD)",
        "Prompt Tokens", "Completion Tokens", "Error",
    ]
    details.append(headers)
    for row in rows:
        details.append([
            row.scenario_id, row.department, row.language, row.stage, row.model_id,
            row.model_label, row.model_slug, row.field, row.expected, row.output,
            row.score, row.justification, row.judge_model, row.latency_sec,
            row.cost_usd, row.prompt_tokens, row.completion_tokens, row.error,
        ])

    summary = workbook.create_sheet("Summary")
    summary.append([
        "Scenario", "Department", "Stage", "Model", "Model Slug",
        "Average Score (0-100)", "Latency (sec)", "Cost (USD)", "Errors",
    ])
    grouped: dict[tuple[str, str, str, str, str], list[EvaluationRow]] = {}
    for row in rows:
        key = (
            row.scenario_id,
            row.department,
            row.stage,
            row.model_label,
            row.model_slug,
        )
        grouped.setdefault(key, []).append(row)
    for key, values in grouped.items():
        latencies = [v.latency_sec for v in values if v.latency_sec is not None]
        costs = [v.cost_usd for v in values if v.cost_usd is not None]
        scores = [v.score for v in values if v.score is not None]
        errors = sorted({v.error for v in values if v.error})
        summary.append([
            *key, (sum(scores) / len(scores) if scores else None),
            max(latencies, default=None), max(costs, default=None),
            " | ".join(errors),
        ])

    errors_sheet = workbook.create_sheet("Errors")
    errors_sheet.append(headers)
    for row in rows:
        if row.error:
            errors_sheet.append([
                row.scenario_id, row.department, row.language, row.stage, row.model_id,
                row.model_label, row.model_slug, row.field, row.expected, row.output,
                row.score, row.justification, row.judge_model, row.latency_sec,
                row.cost_usd, row.prompt_tokens, row.completion_tokens, row.error,
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
@pytest.mark.skipif(
    not RUN_EVALUATION,
    reason="Set RUN_REAL_USE_CASE_EVAL=1 to run live model evaluation",
)
async def test_real_use_case_models_generate_report():
    """Evaluate all configured extraction/reasoning models and write XLSX output."""
    if not WORKBOOK_PATH.exists():
        pytest.fail(f"Workbook not found: {WORKBOOK_PATH}")

    scenarios = load_scenarios()
    if SCENARIO_FILTER:
        requested = {
            item.strip().upper()
            for item in SCENARIO_FILTER.split(",")
            if item.strip()
        }
        scenarios = [
            scenario for scenario in scenarios
            if scenario.scenario_id in requested
        ]
        if not scenarios:
            pytest.fail(
                f"No matching scenario for REAL_USE_CASE_SCENARIO="
                f"{SCENARIO_FILTER!r}"
            )
    else:
        assert len(scenarios) == 6, (
            f"Expected six scenarios, found {len(scenarios)}"
        )
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
                    rows.extend(
                        _rows_for_extraction(scenario, model, None, meta, str(exc))
                    )

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
                    rows.extend(
                        _rows_for_reasoning(scenario, model, None, meta, str(exc))
                    )
    finally:
        await runtime.close()

    judge_runtime = LLMRuntime()
    try:
        await _score_rows(judge_runtime, rows)
    finally:
        await judge_runtime.close()

    _write_report(rows, REPORT_PATH)
    print(f"Wrote {len(rows)} result rows to {REPORT_PATH}")
    errors = sorted({row.error for row in rows if row.error})
    if FAIL_ON_ERROR and errors:
        pytest.fail("Model evaluation errors:\n" + "\n".join(errors))
