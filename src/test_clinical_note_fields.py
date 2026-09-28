"""
test_clinical_note_fields.py
=============================

Runs the real extraction pipeline (RefineText -> Translate -> ExtractFeature)
against every scenario dictated in
`data/Medical_Voice_Assistant_Expected_Fields_Expanded.xlsx`, and scores each
extracted field against the sheet — BOTH with patient history off and with
it on. OPD-4 is intentionally excluded (it's a "patient code not found"
negative control per the sheet's own description — nothing new to check
there beyond OPD-3's negative-control coverage).

For each scenario this file:
  1. Runs refine -> translate ONCE (identical for both passes).
  2. Runs ExtractFeature.extract ONCE with patient_history_text="" (History
     OFF) and grades every field against the "Expected — History OFF"
     column.
  3. Looks up the scenario's Patient Code via model.patient_history
     (get_patient_visits + format_history_for_prompt — the same functions
     app.py's real /audio/analyze endpoint uses), runs ExtractFeature.extract
     a second time with that history text (History ON), and grades every
     field against the "Expected — History ON" column, the exact same way
     the OFF pass is graded.

     NOTE: the workbook's "Should Differ?" column is intentionally NOT
     used anywhere in this file. Earlier versions branched scoring on it
     (grading Should-Differ=No fields against the OFF run's own output as
     a "stability" check instead of the ON column). That's gone — every
     field, OFF and ON alike, is just graded against its own column's
     expected text. Simpler, and avoids the ON pass ever being graded
     against a stand-in value instead of the sheet's actual expectation.

     Also runs model.patient_history.check_history_usage() (if available)
     as a soft, non-scored diagnostic — cross-checking the model's own
     history_considered/history_notes claim against actual ICD-10 family
     overlap with the patient's real history — and prints its flag.
  4. Reports OFF%, ON%, and a combined % per scenario, plus a full summary
     at the end of the session.

SCORING METHODOLOGY
--------------------
The workbook's own "Source" column tells us how strictly to grade content,
so we lean on it instead of inventing our own rubric:

  * Source contains "not mentioned" / "not applicable" / "n/a" / similar,
    OR the expected text itself IS (ignoring parenthetical asides) one of
    "blank" / "none" / "n/a" / "empty" / "nil", possibly combined with
    "/" or "or" (e.g. "None / blank", "N/A (new complaint)")
        -> the field is expected to be BLANK-OR-EQUIVALENT (and, per the
           app's own design, should surface in QuestionGenerator's
           questions instead). Score 1.0 if the actual value is itself
           blank or one of those same equivalent words (a literal "None"
           counts the same as an empty string here — it shouldn't be
           punished for spelling out what an empty field implies). Score
           0.2 if the model populated it with real content anyway (a real
           miss, not graded 0 because an over-eager-but-not-wrong
           extraction is a lesser sin than a wrong one).

  * Source contains "ambiguous"
        -> the sheet itself says the correct answer could reasonably be
           blank OR one of several described alternatives (e.g. "may
           duplicate Onset, or be left blank"). Score 1.0 if blank.
           Otherwise this is handed to the LLM judge with an explicit
           grading note describing the acceptable-answer space.

  * icd10_codes / final_diagnosis specifically
        -> graded deterministically by ICD-10 code structure, not text
           similarity: exact code match (ignoring the 7th-character
           encounter-type suffix, e.g. the trailing "A" in "S61.409A")
           scores 1.0; a matching category/"parent" (first 3 characters,
           e.g. "S61") with a different subcode scores 0.70 ("right body
           region/type, wrong specific code"); no category overlap scores
           0.1. Blended 85/15 with light lexical similarity on the full
           text so a badly-worded but correctly-coded description isn't
           scored identically to a perfectly-worded one. Falls back to the
           standard content grading below if no ICD-10-shaped code is
           found in the expected text at all.

  * Everything else ("Stated", "AI-recommended" prose fields like
    final_diagnosis_rationale, "History field", etc.)
        -> 0.0 if the model left it blank, otherwise the LLM judge's
           semantic/clinical-equivalence score (0-100, averaged across
           CLINICAL_JUDGE_RUNS re-judgings — see "JUDGE BIAS MITIGATION"
           below) is used directly, with NO lexical/token-overlap penalty.
           An earlier version blended in text-similarity here, which
           unfairly punished correct answers phrased or notated
           differently from the sheet (e.g. "FDI 46" vs "46" for the same
           tooth) — the sheet's own grading note says to grade CONTENT,
           not wording, so wording differences alone no longer cost points.

A scenario's overall score is the mean of ALL its field scores across both
passes. The pass threshold is intentionally soft (LLM output varies run to
run) — tune it with CLINICAL_NOTE_PASS_THRESHOLD.

SETUP
-----
1. Run pytest from a working directory where `model.*` and `core.config`
   are importable (i.e. your project root) — this file imports the real
   pipeline modules, it does not reimplement them.
2. Make sure `core.config.Config.OPENROUTER_API_KEY` (or the
   OPENROUTER_API_KEY / OPENROUTER_API_KEYS env vars it reads) is set —
   this suite makes real OpenRouter calls, both through your pipeline and
   through the LLM judge.
3. Make sure the patient-history data `model.patient_history` reads from
   ("Visit Level History.csv", by default at <repo_root>/data/, or wherever
   Config.PATIENT_HISTORY_CSV_PATH points) is present and contains the
   sample Patient Codes the workbook references — otherwise every History
   ON pass will find zero prior visits and degenerate to the OFF case (the
   test still runs and reports this, it just won't be a meaningful
   History-ON check for that scenario).
4. Put the workbook at tests/data/Medical_Voice_Assistant_Expected_Fields_Expanded.xlsx
   relative to this file, or point CLINICAL_NOTES_XLSX at it.

ENV VARS
--------
  CLINICAL_NOTES_XLSX            path to the workbook (see above)
  CLINICAL_NOTE_PASS_THRESHOLD   float 0-1, default 0.65
  CLINICAL_JUDGE_MODEL           OpenRouter model id for the LLM judge,
                                  default "openai/gpt-6-luna"
  CLINICAL_JUDGE_RUNS            how many times to re-judge each scenario's
                                  batch (each with the fields re-shuffled),
                                  averaging the per-field scores — see
                                  "JUDGE BIAS MITIGATION" below. Default 3;
                                  set to 1 to disable (cheaper, noisier)
  CLINICAL_NOTES_SKIP            extra comma-separated scenario IDs to
                                  skip, on top of OPD-4 which is always
                                  skipped
  CLINICAL_NOTES_TEST_HISTORY_ON set to "0"/"false" to run History-OFF
                                  only (skips patient-history lookup and
                                  the second extract() call entirely)

JUDGE BIAS MITIGATION
----------------------
Three concrete steps, matched to the biases that actually apply to this
judge's task (scoring one candidate value against a reference value — NOT
ranking two peer responses against each other, which is where most
"LLM-as-judge" bias literature is written for):

  * Verbosity bias — the prompt explicitly instructs the judge not to
    reward length or penalize brevity: a short correct answer must score
    identically to a verbose one stating the same facts, and padding a
    response with restated or irrelevant detail must not raise its score.

  * List-position bias — every scenario's judged fields are sent to the
    judge as one batched prompt (see _judge_batch), and a field's position
    in that list could otherwise correlate with how carefully it gets
    graded (a documented effect in batched-LLM-grading setups). This file
    re-sends the batch CLINICAL_JUDGE_RUNS times (default 3), reshuffling
    the field order each time, and averages each field's score across the
    runs — see _judge_batch_multi. This also smooths out ordinary
    run-to-run judge noise as a side effect. If a field's scores vary
    widely across runs (default: a >15-point spread), its verdict is
    prefixed with a "[judge variance ...]" flag so that's visible in the
    report rather than silently averaged away.

  * Classic A/B positional bias (favoring whichever candidate is shown
    first) does NOT apply here and is intentionally not "fixed" by
    swapping which text is labeled "expected" vs "actual": those two
    roles are not interchangeable peers, "expected" is always the
    reference the sheet defines as correct and "actual" is always the
    model's output being graded, so swapping them would ask a different,
    meaningless question rather than cancel a bias.

  * Self-enhancement/model-family bias — a judge model can rate output
    from a same-provider/same-family model more favorably. CLINICAL_JUDGE_MODEL
    is a plain setting for exactly this reason: if your extraction
    pipeline's model is also from OpenAI, consider pointing the judge at
    a different provider/family to rule this out, since this file has no
    way to detect or correct for it automatically.

USAGE
-----
  pytest test_clinical_note_fields.py -s                  # full run, verbose report
  pytest test_clinical_note_fields.py -s -k "ER-1"         # one scenario
  pytest test_clinical_note_fields.py -s -k "not Dental"   # skip a department
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openpyxl
import pytest

# ── Real pipeline modules — this file never reimplements pipeline logic ──
RefineText = pytest.importorskip(
    "model.refine_text",
    reason="Run pytest from the project root so model.refine_text is importable.",
).RefineText
Translate = pytest.importorskip("model.translation").Translate
ExtractFeature = pytest.importorskip("model.extract_features").ExtractFeature
_dept_fields_mod = pytest.importorskip("model.department_fields")
get_department_fields = _dept_fields_mod.get_department_fields
Config = pytest.importorskip("core.config").Config

TEST_HISTORY_ON = os.environ.get("CLINICAL_NOTES_TEST_HISTORY_ON", "1").strip().lower() not in ("0", "false", "no")

if TEST_HISTORY_ON:
    _patient_history_mod = pytest.importorskip(
        "model.patient_history",
        reason=(
            "model.patient_history isn't importable — needed for the History-ON pass. "
            "Set CLINICAL_NOTES_TEST_HISTORY_ON=0 to run History-OFF only instead."
        ),
    )
    get_patient_visits = _patient_history_mod.get_patient_visits
    format_history_for_prompt = _patient_history_mod.format_history_for_prompt
    check_history_usage = getattr(_patient_history_mod, "check_history_usage", None)


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

XLSX_PATH = Path(
    os.environ.get(
        "CLINICAL_NOTES_XLSX",
        "/home/ai/Workspace/Doha/MedVoice/src/data/Medical_Voice_Assistant_Expected_Fields_Expanded.xlsx",
    )
)

PASS_THRESHOLD = float(os.environ.get("CLINICAL_NOTE_PASS_THRESHOLD", "0.65"))
JUDGE_MODEL = os.environ.get("CLINICAL_JUDGE_MODEL", "openai/gpt-6-luna")
# Bias mitigation: each scenario's judged fields are graded this many times,
# each with the field list presented in a freshly randomized order, and the
# per-field scores are averaged — see _judge_batch_multi for why.
JUDGE_RUNS = max(1, int(os.environ.get("CLINICAL_JUDGE_RUNS", "3")))

# OPD-4 is a "patient code not found" negative control that this file
# intentionally does not run (see module docstring). Always excluded.
SKIP_SCENARIOS = {"OPD-4"} | {
    s.strip().upper() for s in os.environ.get("CLINICAL_NOTES_SKIP", "").split(",") if s.strip()
}

SCENARIO_SHEETS = ["ER Scenarios", "OPD Scenarios", "Dental Scenarios"]
SCENARIO_ID_RE = re.compile(r"^([A-Za-z]+-\d+)\b")
PATIENT_CODE_RE = re.compile(r"\d+")

# Extraction-JSON keys for the "always AI-recommended" fields that live
# outside model.department_fields.DEPARTMENT_FIELDS (see extract_features.py
# / department_fields.py docstrings for why they're excluded from there).
EXTRA_FIELDS = {
    "triage_priority": "Triage Priority",
    "icd10_codes": "ICD-10 Codes",
    "final_diagnosis": "Final Diagnosis",
    "final_diagnosis_rationale": "Final Diagnosis Rationale",
    "history_considered": "history_considered",
    "history_notes": "history_notes",
}

# Fields graded by ICD-10 code structure (parent/child match) instead of
# plain text similarity — see module docstring.
ICD10_FIELDS = {"icd10_codes", "final_diagnosis"}

# Fields intentionally left out of scoring entirely (not graded, not
# included in a scenario's OFF/ON/combined percentage, not printed in the
# per-field report). They're still parsed from the sheet — just dropped
# when a Scenario's expected_fields is built — so re-including one later
# is a one-line change here, not a re-plumb.
EXCLUDED_FROM_SCORING = {"final_diagnosis_rationale", "history_considered", "history_notes"}

if not XLSX_PATH.exists():
    pytest.skip(
        f"Expected-fields workbook not found at {XLSX_PATH}. "
        f"Set CLINICAL_NOTES_XLSX or place the file at tests/data/... next to this test file.",
        allow_module_level=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Workbook parsing
# ═══════════════════════════════════════════════════════════════════════

def _norm_label(label: str) -> str:
    """Case/dash/punctuation-insensitive label key, so 'A – Airway' (en
    dash, in code) matches 'A — Airway' (em dash, in the sheet), and
    'SubSpecialty Focus' (code) matches 'Sub-specialty Focus' (sheet)."""
    return re.sub(r"[^a-z0-9]", "", (label or "").lower())


def _build_label_key_map(department: str) -> dict[str, str]:
    m = {}
    for f in get_department_fields(department):
        m[_norm_label(f["label"])] = f["key"]
    for key, label in EXTRA_FIELDS.items():
        m[_norm_label(label)] = key
    return m


def _parse_scripts(wb) -> dict[str, dict]:
    ws = wb["Scripts"]
    rows = list(ws.iter_rows(values_only=True))
    header, data_rows = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    out = {}
    for row in data_rows:
        if not row or not row[0]:
            continue
        scenario_id = str(row[0]).strip()
        department = str(row[idx["Department"]]).strip().lower()
        mode = str(row[idx["Mode"]]).strip().lower()
        language_raw = str(row[idx["Language"]]).strip().lower()
        specialty_raw = (row[idx["Specialty"]] or "").strip()
        patient_code_raw = str(row[idx["Patient Code"]]).strip()
        code_match = PATIENT_CODE_RE.search(patient_code_raw)
        script = row[idx["Dictation Script (identical for both History OFF and History ON runs)"]]
        out[scenario_id] = {
            "department": department,
            "is_conversation": mode == "conversation",
            "language": "ar" if language_raw.startswith("ar") else "en",
            "specialty": None if specialty_raw.lower().startswith("general") or not specialty_raw else specialty_raw,
            "patient_code": code_match.group(0) if code_match else None,
            "script": script,
        }
    return out


@dataclass
class FieldSpec:
    source: str
    expected_off: str
    expected_on: str
    notes: str = ""


def _parse_scenario_sheet(ws) -> dict[str, dict[str, FieldSpec]]:
    """Returns {scenario_id: {field_label: FieldSpec}}."""
    rows = list(ws.iter_rows(values_only=True))
    scenarios: dict[str, dict[str, FieldSpec]] = {}
    current_id: Optional[str] = None

    for row in rows:
        first = row[0]
        # Title row: e.g. "ER-1 — Nose Wound Recurrence (...)" with every
        # other cell empty.
        if isinstance(first, str) and row[1] is None:
            m = SCENARIO_ID_RE.match(first.strip())
            if m:
                current_id = m.group(1)
                scenarios.setdefault(current_id, {})
            continue
        if first == "Field" and row[1] == "Source":
            continue  # sub-table header row
        if current_id and first and row[1]:
            label = str(first).strip()
            source = str(row[1]).strip()
            expected_off = "" if row[2] is None else str(row[2]).strip()
            expected_on = "" if row[3] is None else str(row[3]).strip()
            notes = "" if row[5] is None else str(row[5]).strip()
            scenarios[current_id][label] = FieldSpec(source, expected_off, expected_on, notes)

    return scenarios


def _load_scenarios(wb) -> dict[str, dict[str, FieldSpec]]:
    merged: dict[str, dict[str, FieldSpec]] = {}
    for sheet_name in SCENARIO_SHEETS:
        merged.update(_parse_scenario_sheet(wb[sheet_name]))
    return merged


# ═══════════════════════════════════════════════════════════════════════
# Scenario model  (named "Scenario", NOT "TestCase" — pytest's default
# collection rules try to collect any class named Test*, which would emit
# a PytestCollectionWarning for a plain dataclass like this one).
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    id: str
    department: str
    is_conversation: bool
    language: str
    specialty: Optional[str]
    script: str
    patient_code: Optional[str]
    # json_key -> FieldSpec
    expected_fields: dict[str, FieldSpec] = field(default_factory=dict)


def _build_scenarios() -> list[Scenario]:
    wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
    scripts = _parse_scripts(wb)
    scenario_fields = _load_scenarios(wb)

    scenarios = []
    for scenario_id, meta in scripts.items():
        if scenario_id in SKIP_SCENARIOS:
            continue
        if scenario_id not in scenario_fields:
            continue
        label_key_map = _build_label_key_map(meta["department"])
        expected_fields: dict[str, FieldSpec] = {}
        for label, spec in scenario_fields[scenario_id].items():
            key = label_key_map.get(_norm_label(label))
            if key is None:
                # Field named in the sheet but not recognized for this
                # department/extraction schema — nothing to compare it to.
                continue
            if key in EXCLUDED_FROM_SCORING:
                continue
            expected_fields[key] = spec

        scenarios.append(
            Scenario(
                id=scenario_id,
                department=meta["department"],
                is_conversation=meta["is_conversation"],
                language=meta["language"],
                specialty=meta["specialty"],
                script=meta["script"],
                patient_code=meta["patient_code"],
                expected_fields=expected_fields,
            )
        )
    scenarios.sort(key=lambda s: s.id)
    return scenarios


ALL_SCENARIOS = _build_scenarios()

if not ALL_SCENARIOS:
    pytest.skip("No scenarios parsed from the workbook — check its sheet layout.", allow_module_level=True)


# ═══════════════════════════════════════════════════════════════════════
# Scoring primitives
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FieldScore:
    key: str
    phase: str  # "OFF" | "ON"
    expected: str
    actual: str
    score: float
    verdict: str


@dataclass
class GradeResult:
    score: float
    verdict: str


def _flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value).strip()


# ── "blank / none / n/a"-equivalence ─────────────────────────────────────

_EMPTY_TOKEN_RE = re.compile(r"^(none|blank|n/a|na|empty|nil)$")
_EMPTY_SPLIT_RE = re.compile(r"\s+/\s+|\s*,\s*|\s+or\s+", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\(.*?\)")


def _is_none_or_blank_text(text: str) -> bool:
    """True for expected/actual text that IS (once any parenthetical
    aside is stripped) nothing but 'none' / 'blank' / 'n/a' / 'empty' /
    'nil', possibly combined with '/' or 'or' — e.g. "None / blank",
    "N/A (new complaint)". Deliberately does NOT match real content that
    merely mentions one of those words in passing."""
    t = _PARENTHETICAL_RE.sub("", (text or "")).strip().lower()
    if not t:
        return False
    parts = [p.strip() for p in _EMPTY_SPLIT_RE.split(t) if p.strip()]
    return bool(parts) and all(_EMPTY_TOKEN_RE.match(p) for p in parts)


def _is_blank_like(text: str) -> bool:
    """True for an empty string OR a none/blank-equivalent value."""
    t = (text or "").strip()
    return (not t) or _is_none_or_blank_text(t)


def _field_bucket(source: str, expected_text: str) -> str:
    s = (source or "").lower()
    not_mentioned_markers = ("not mentioned", "not applicable", "n/a", "not explicitly asked", "not clearly stated")
    if any(marker in s for marker in not_mentioned_markers):
        return "empty"
    if _is_none_or_blank_text(expected_text):
        return "empty"
    if "ambiguous" in s:
        return "ambiguous"
    return "standard"


def _lexical_score(expected: str, actual: str) -> float:
    import difflib

    def norm(t: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()

    e, a = norm(expected), norm(actual)
    return difflib.SequenceMatcher(None, e, a).ratio() if (e or a) else 1.0


# ── ICD-10 parent/child code grading ─────────────────────────────────────

_ICD10_CODE_RE = re.compile(r"\b[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?\b")


def _extract_icd10_codes(text: str) -> list[str]:
    return _ICD10_CODE_RE.findall((text or "").upper())


def _icd10_root(code: str) -> str:
    """Code without the optional 7th-character encounter-type suffix
    (e.g. "S61.409A" -> "S61.409"), so "initial encounter" vs "subsequent
    encounter" coding of an otherwise-identical code still counts as an
    exact match."""
    c = code.upper()
    if len(c) > 3 and c[-1].isalpha():
        return c[:-1]
    return c


def _icd10_family(code: str) -> str:
    """The category/"parent" — letter + 2 digits, e.g. "S61" from
    "S61.409A", mirroring the same code-family logic model.patient_history
    uses for its own history-overlap cross-check."""
    return code.upper()[:3]


def _grade_icd10_field(expected_text: str, actual_text: str) -> Optional[GradeResult]:
    """Deterministic parent/child ICD-10 grading. Returns None (caller
    should fall back to standard content grading) if the expected text
    doesn't contain anything code-shaped to compare against."""
    expected_codes = _extract_icd10_codes(expected_text)
    if not expected_codes:
        return None
    actual_codes = _extract_icd10_codes(actual_text)
    if not actual_codes:
        return GradeResult(0.0, "no ICD-10 code found in actual output")

    exp_roots, act_roots = {_icd10_root(c) for c in expected_codes}, {_icd10_root(c) for c in actual_codes}
    exp_fams, act_fams = {_icd10_family(c) for c in expected_codes}, {_icd10_family(c) for c in actual_codes}

    exact = exp_roots & act_roots
    if exact:
        code_result = GradeResult(1.0, f"exact ICD-10 code match ({', '.join(sorted(exact))})")
    else:
        family_overlap = exp_fams & act_fams
        if family_overlap:
            code_result = GradeResult(
                0.70, f"same ICD-10 category ({', '.join(sorted(family_overlap))}) — right region/type, wrong subcode"
            )
        else:
            code_result = GradeResult(
                0.1, f"different ICD-10 category — expected {sorted(exp_fams)}, got {sorted(act_fams)}"
            )

    lex = _lexical_score(expected_text, actual_text)
    blended = 0.85 * code_result.score + 0.15 * lex
    return GradeResult(blended, code_result.verdict)


def _content_grade(key: str, source: str, expected_text: str, actual_val: str) -> Optional[GradeResult]:
    """Returns a GradeResult if the score is determinable without an LLM
    call (blank-equivalent fields, missing content, ICD-10 code
    comparisons), else None — meaning the caller should queue this field
    for the batch LLM judge."""
    bucket = _field_bucket(source, expected_text)
    if bucket == "empty":
        if _is_blank_like(actual_val):
            return GradeResult(1.0, "blank (or equivalent) as expected")
        return GradeResult(0.2, "populated a field that shouldn't be")
    if bucket == "ambiguous" and not actual_val:
        return GradeResult(1.0, "blank — acceptable (ambiguous)")
    if bucket == "standard" and not actual_val:
        return GradeResult(0.0, "missing — expected content")
    if key in ICD10_FIELDS and actual_val:
        icd_result = _grade_icd10_field(expected_text, actual_val)
        if icd_result is not None:
            return icd_result
    return None


def _judge_hint(bucket: str, expected_text: str) -> str:
    if bucket == "ambiguous":
        return (
            f'This field\'s ground truth is AMBIGUOUS — acceptable answers: "{expected_text}". '
            f"Score close to 100 if the actual value is consistent with ANY of the described acceptable "
            f"answers (including a short value that duplicates another field, e.g. onset/duration), even "
            f"if brief. Don't penalize for brevity or for not repeating the instructional wording."
        )
    return ""


def _judge_client():
    from openai import OpenAI

    api_key = getattr(Config, "OPENROUTER_API_KEY", None) or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("No OPENROUTER_API_KEY configured — cannot run the pipeline or the LLM judge.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def _safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


def _judge_batch(client, scenario_id: str, items: list[dict]) -> dict:
    """One LLM call scoring every field that needs semantic judgment for a
    scenario/phase, instead of one call per field. Items may carry a
    "hint" (see _judge_hint) describing the acceptable-answer space for
    ambiguous fields. Items are graded in the order given — callers doing
    repeated/shuffled sampling (see _judge_batch_multi) pass a
    pre-shuffled list."""
    if not items:
        return {}

    numbered = "\n\n".join(
        f'{i}. field_id: "{it["id"]}"\n   expected: "{it["expected"]}"\n   actual: "{it["actual"]}"'
        + (f'\n   grading_note: {it["hint"]}' if it.get("hint") else "")
        for i, it in enumerate(items)
    )
    prompt = f"""You are grading a medical scribe AI's extracted clinical-note fields
against a human-written reference answer, for scenario {scenario_id}.

Grade on CLINICAL CONTENT AND MEANING ONLY — never penalize for wording,
phrasing, notation, formatting, or field placement. Paraphrases, reordering,
different but equivalent notations (e.g. "FDI 46" vs "46" for the same
tooth), and reasonable synonyms should score 100 if they preserve the same
clinical facts. Penalize missing facts, contradicted facts, or fabricated
facts not supported by the reference — not superficial differences in how
the same fact is expressed. Where a "grading_note" is present, follow it —
it overrides the general instructions above for that field.

Do not let response length affect the score either way: a short answer
that states the same facts as the reference must score exactly as high as
a longer one, and a longer answer that pads with restated, redundant, or
clinically irrelevant detail must NOT score higher than a concise one
covering the same facts. Judge only whether the correct clinical content
is present — never how much text was used to say it.

Fields to grade:

{numbered}

Respond with ONLY a JSON object mapping each field_id to an object with:
  "score": integer 0-100 (100 = fully equivalent content, 0 = wrong/contradictory)
  "verdict": a short (<12 word) explanation

Example shape: {{"chief_complaint": {{"score": 92, "verdict": "same complaint, different wording"}}}}
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content if response.choices else "{}"
    try:
        return _safe_parse_json(content)
    except (ValueError, json.JSONDecodeError):
        return {}


# How far apart a field's min/max score across judge runs has to be before
# it's flagged as unstable in the report, rather than silently averaged.
JUDGE_VARIANCE_FLAG_THRESHOLD = 15


def _judge_batch_multi(client, scenario_id: str, items: list[dict], runs: int = JUDGE_RUNS) -> dict:
    """Runs _judge_batch `runs` times, reshuffling the field order each
    time, and averages each field's score across runs — see "JUDGE BIAS
    MITIGATION" in the module docstring for why. Returns the same shape as
    _judge_batch: {field_id: {"score": 0-100 average, "verdict": str}},
    with the verdict prefixed with a variance flag when the field's scores
    disagreed widely across runs."""
    if not items:
        return {}

    per_field_scores: dict[str, list[float]] = {it["id"]: [] for it in items}
    per_field_verdicts: dict[str, list[str]] = {it["id"]: [] for it in items}

    for _ in range(runs):
        shuffled = items[:]
        random.shuffle(shuffled)
        result = _judge_batch(client, scenario_id, shuffled)
        for it in items:
            obj = result.get(it["id"], {})
            per_field_scores[it["id"]].append(float(obj.get("score", 50)))
            per_field_verdicts[it["id"]].append(obj.get("verdict", "no judge response"))

    merged: dict[str, dict] = {}
    for field_id, scores in per_field_scores.items():
        avg = sum(scores) / len(scores)
        spread = max(scores) - min(scores)
        # Pick the verdict from whichever run landed closest to the average
        # as the representative explanation.
        closest_idx = min(range(len(scores)), key=lambda i: abs(scores[i] - avg))
        verdict = per_field_verdicts[field_id][closest_idx]
        if spread > JUDGE_VARIANCE_FLAG_THRESHOLD:
            verdict = f"[judge variance {min(scores):.0f}-{max(scores):.0f}% across {runs} runs] {verdict}"
        merged[field_id] = {"score": avg, "verdict": verdict}
    return merged


# ═══════════════════════════════════════════════════════════════════════
# Per-scenario scoring: History OFF + History ON
# ═══════════════════════════════════════════════════════════════════════

def score_scenario(scenario: Scenario, off_json: dict, on_json: Optional[dict], client) -> list[FieldScore]:
    """Grades a scenario's OFF pass, and (if on_json is not None) its ON
    pass, against the workbook. Every field in every phase is graded
    against its OWN column's expected text ("Expected — History OFF" for
    the OFF pass, "Expected — History ON" for the ON pass) — the sheet's
    "Should Differ?" column is not consulted. See module docstring for the
    full rubric."""
    results: list[FieldScore] = []
    to_judge: list[dict] = []  # {id, expected, actual, hint} batch for the judge

    def _grade_phase(key: str, source: str, expected_text: str, actual_val: str, phase: str) -> None:
        pre = _content_grade(key, source, expected_text, actual_val)
        if pre is not None:
            results.append(FieldScore(key, phase, expected_text, actual_val, pre.score, pre.verdict))
            return
        bucket = _field_bucket(source, expected_text)
        to_judge.append({
            "id": f"{key}::{phase}",
            "expected": expected_text,
            "actual": actual_val,
            "hint": _judge_hint(bucket, expected_text),
        })

    for key, spec in scenario.expected_fields.items():
        off_val = _flatten(off_json.get(key))
        _grade_phase(key, spec.source, spec.expected_off, off_val, "OFF")

        if on_json is not None:
            on_val = _flatten(on_json.get(key))
            _grade_phase(key, spec.source, spec.expected_on, on_val, "ON")

    judged = _judge_batch_multi(client, scenario.id, to_judge) if to_judge else {}
    for item in to_judge:
        key, phase = item["id"].split("::")
        obj = judged.get(item["id"], {})
        score = float(obj.get("score", 50)) / 100.0
        verdict = obj.get("verdict", "no judge response — defaulted to 50%")
        results.append(FieldScore(key, phase, item["expected"], item["actual"], score, verdict))

    results.sort(key=lambda r: (r.key, r.phase))
    return results


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def _print_scenario_report(
    scenario: Scenario, scores: list[FieldScore], off_pct: float, on_pct: Optional[float], combined: float,
    history_check: Optional[dict] = None, visit_count: Optional[int] = None,
) -> None:
    print(f"\n{'=' * 78}")
    header = f"{scenario.id} ({scenario.department.upper()}, {scenario.language}, {'conversation' if scenario.is_conversation else 'doctor'})"
    if on_pct is not None:
        header += f"  —  OFF: {off_pct:.1%}  ON: {on_pct:.1%}  combined: {combined:.1%}"
    else:
        header += f"  —  OFF: {off_pct:.1%} (History-ON not run)"
    print(header)
    if visit_count is not None:
        print(f"  patient_code={scenario.patient_code!r}  prior visits found: {visit_count}")
    if history_check is not None:
        flag = history_check.get("flag")
        note = f"  [history cross-check] flag={flag!r} overlap_families={history_check.get('overlap_families')}"
        print(note if flag else f"  [history cross-check] ok — claimed={history_check.get('claimed_considered')} overlap={history_check.get('overlap_found')}")
    print(f"{'-' * 78}")
    for s in scores:
        flag = "OK  " if s.score >= 0.8 else ("~   " if s.score >= 0.5 else "FAIL")
        print(f"  [{flag}] {s.score:5.0%}  [{s.phase}] {s.key:<28} {s.verdict}")
        if s.score < 0.8:
            print(f"           expected: {s.expected[:90]}")
            print(f"           actual:   {s.actual[:90] or '(blank)'}")
    print(f"{'=' * 78}\n")


@pytest.fixture(scope="session")
def judge_client():
    return _judge_client()


@pytest.fixture(scope="session")
def session_results():
    results: list[tuple[str, float, Optional[float], float]] = []  # id, off, on, combined
    yield results
    if not results:
        return
    print(f"\n{'#' * 78}")
    print("CLINICAL NOTE FIELD EXTRACTION — SUMMARY")
    print(f"{'#' * 78}")
    for scenario_id, off_pct, on_pct, combined in results:
        bar = "#" * int(combined * 40)
        on_str = f"ON {on_pct:5.1%}" if on_pct is not None else "ON   n/a"
        print(f"  {scenario_id:<10} OFF {off_pct:5.1%}  {on_str}  COMBINED {combined:5.1%}  {bar}")
    avg = sum(c for _, _, _, c in results) / len(results)
    print(f"{'-' * 78}")
    print(f"  {'AVERAGE':<10} {'':20} {'':13} COMBINED {avg:5.1%}")
    print(f"{'#' * 78}\n")


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.llm
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s.id for s in ALL_SCENARIOS])
def test_clinical_note_extraction(scenario: Scenario, judge_client, session_results):
    refined = RefineText.refining_transcription(
        scenario.script, scenario.language, is_conversation=scenario.is_conversation, specialty=scenario.specialty
    )
    translated = Translate.translate(refined, is_conversation=scenario.is_conversation)

    off_json, _ = ExtractFeature.extract(
        translated, department=scenario.department, patient_history_text="", is_conversation=scenario.is_conversation
    )
    off_json = off_json or {}

    on_json = None
    history_check = None
    visit_count = None
    if TEST_HISTORY_ON:
        history_text = ""
        visits = []
        if scenario.patient_code:
            visits = get_patient_visits(scenario.patient_code) or []
            history_text = format_history_for_prompt(visits) or ""
        visit_count = len(visits)
        if visit_count == 0:
            print(f"\n[{scenario.id}] no prior visits found for patient code {scenario.patient_code!r} "
                  f"— History-ON pass will behave like History-OFF.")
        on_json, _ = ExtractFeature.extract(
            translated, department=scenario.department, patient_history_text=history_text, is_conversation=scenario.is_conversation
        )
        on_json = on_json or {}
        if history_text and check_history_usage is not None:
            try:
                history_check = check_history_usage(visits, on_json)
            except Exception as exc:  # diagnostic only — never fail the test over this
                print(f"[{scenario.id}] check_history_usage raised {exc!r} — skipping that diagnostic")

    scores = score_scenario(scenario, off_json, on_json, judge_client)
    off_scores = [s for s in scores if s.phase == "OFF"]
    on_scores = [s for s in scores if s.phase == "ON"]
    off_pct = sum(s.score for s in off_scores) / len(off_scores) if off_scores else 0.0
    on_pct = (sum(s.score for s in on_scores) / len(on_scores)) if on_scores else None
    combined = sum(s.score for s in scores) / len(scores) if scores else 0.0

    session_results.append((scenario.id, off_pct, on_pct, combined))
    _print_scenario_report(scenario, scores, off_pct, on_pct, combined, history_check, visit_count)

    failing = [s for s in scores if s.score < 0.5]
    detail = "\n".join(f"  - [{s.phase}] {s.key}: {s.score:.0%} ({s.verdict})" for s in failing)
    assert combined >= PASS_THRESHOLD, (
        f"{scenario.id} scored {combined:.1%} combined, below threshold {PASS_THRESHOLD:.0%}.\n"
        f"Worst fields:\n{detail}"
    )