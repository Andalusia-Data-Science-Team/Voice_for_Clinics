"""
test_clinical_note_fields.py
=============================

Runs the real extraction pipeline (RefineText -> Translate -> ExtractFeature)
against every scenario dictated in `real_use_cases.xlsx`, and scores each
extracted field against the sheet.

*** SCHEMA CHANGE ***
This workbook is shaped differently from the original expanded workbook and
this file has been rewritten around it:

  * `Scripts` sheet is only 3 columns: Departement | Language | Scenario.
    There is no Mode, Specialty, or Patient Code column, and no scenario-ID
    column — scenario IDs ("ER-1", "OPD-1", "Dental-1", ...) are inferred by
    counting non-blank rows per department, in the order they appear, and
    matched positionally against the scenario sheets below.

  * Each of "ER Scenarios" / "OPD Scenarios" / "Dental Scenarios" is a plain
    Field | Expected sub-table per scenario (no Source column, no separate
    History-OFF/History-ON expected columns, no notes column).

  * Because there's no Patient Code column, there is nothing to look up in
    model.patient_history, so the History-ON pass has been REMOVED entirely.
    This file now does a single extraction pass per scenario
    (patient_history_text=""), equivalent to the old "History OFF" run.

  * Because there's no Source column, the old "not mentioned / ambiguous"
    grading buckets driven by that column are gone. A field is graded as
    "expected blank" purely from its own Expected text being blank or a
    none/blank-equivalent word (see _is_blank_like); otherwise it's graded
    as standard content (LLM judge, or deterministic ICD-10 grading where
    applicable).

  * Mode (conversation vs. dictation) and Specialty aren't in this sheet
    either. All scenarios are treated as dictation (is_conversation=False)
    and general specialty (specialty=None) — the scripts read as narrated
    case presentations, not back-and-forth dialogue. Adjust DEFAULT_IS_
    CONVERSATION below if that assumption is wrong for some department.

SCHEMA-DRIFT WARNING
---------------------
Field labels in this workbook don't necessarily match
model.department_fields.get_department_fields(...)'s labels verbatim (e.g.
this sheet says "Triage Classification" where the pipeline's EXTRA_FIELDS
constant used to say "Triage Priority" — aliased below via LABEL_ALIASES,
but there may be others we don't know about without seeing department_
fields.py). Any Expected-sheet label that can't be matched to a JSON key is
now printed as a warning at collection time (see _build_scenarios) instead
of being silently dropped — check that output the first time you run this
against the new sheet, and add more entries to LABEL_ALIASES as needed.

SCORING METHODOLOGY
--------------------
  * Expected text is blank, or (ignoring parenthetical asides and a
    trailing period) is nothing but "blank" / "none" / "n/a" / "empty" /
    "nil" (possibly combined with "/" or "or", e.g. "None / blank"), OR a
    whole-string "nothing to report" phrase such as "Not specified." or
    "No neurological abnormality reported" (see _EMPTY_PHRASE_RE)
        -> the field is expected to be BLANK-OR-EQUIVALENT. Score 1.0 if
           the actual value is itself blank or one of those same
           equivalent words/phrases. Score 0.2 if the model populated it
           with real content anyway (a real miss, not graded 0 because an
           over-eager-but-not-wrong extraction is a lesser sin than a
           wrong one). A phrase that carries real content beyond "nothing
           here" — e.g. "No known drug allergies", "Not applicable due to
           ICU admission." — is NOT swept into this bucket; see
           _EMPTY_PHRASE_RE's comment for why.

  * icd10_codes / final_diagnosis specifically
        -> graded deterministically by ICD-10 code structure when the
           Expected text actually contains code-shaped tokens (e.g.
           "S61.409A"): exact match (ignoring the trailing encounter-type
           letter) scores 1.0, matching category/parent (first 3 chars)
           scores 0.70, no overlap scores 0.1, blended 85/15 with light
           lexical similarity. In THIS workbook these fields are usually
           free-text diagnosis names rather than actual codes (e.g. "Acute
           coronary syndrome..."), in which case no code-shaped token is
           found and it falls back to standard content grading below.

  * Everything else
        -> 0.0 if the model left it blank, otherwise the LLM judge's
           semantic/clinical-equivalence score (0-100, averaged across
           CLINICAL_JUDGE_RUNS re-judgings — see "JUDGE BIAS MITIGATION"
           below) is used directly, with NO lexical/token-overlap penalty.
           Grade CONTENT, not wording.

A scenario's overall score is the mean of its field scores. The pass
threshold is intentionally soft (LLM output varies run to run) — tune it
with CLINICAL_NOTE_PASS_THRESHOLD.

SETUP
-----
1. Run pytest from a working directory where `model.*` and `core.config`
   are importable (i.e. your project root) — this file imports the real
   pipeline modules, it does not reimplement them.
2. Make sure `core.config.Config.OPENROUTER_API_KEY` (or the
   OPENROUTER_API_KEY / OPENROUTER_API_KEYS env vars it reads) is set —
   this suite makes real OpenRouter calls, both through your pipeline and
   through the LLM judge.
3. Put the workbook at tests/data/real_use_cases.xlsx relative to this
   file, or point CLINICAL_NOTES_XLSX at it.

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
  CLINICAL_NOTES_SKIP            comma-separated scenario IDs to skip

JUDGE BIAS MITIGATION
----------------------
  * Verbosity bias — the prompt explicitly instructs the judge not to
    reward length or penalize brevity.

  * List-position bias — every scenario's judged fields are sent to the
    judge as one batched prompt (see _judge_batch), and a field's position
    in that list could otherwise correlate with how carefully it gets
    graded. This file re-sends the batch CLINICAL_JUDGE_RUNS times
    (default 3), reshuffling the field order each time, and averages each
    field's score across the runs — see _judge_batch_multi. If a field's
    scores vary widely across runs (default: a >15-point spread), its
    verdict is prefixed with a "[judge variance ...]" flag.

  * Classic A/B positional bias (favoring whichever candidate is shown
    first) does NOT apply here and is intentionally not "fixed" by
    swapping which text is labeled "expected" vs "actual": "expected" is
    always the sheet's reference answer, "actual" is always the model's
    output being graded — swapping them would ask a different, meaningless
    question rather than cancel a bias.

  * Self-enhancement/model-family bias — CLINICAL_JUDGE_MODEL is a plain
    setting for exactly this reason: if your extraction pipeline's model
    is also from OpenAI, consider pointing the judge at a different
    provider/family to rule this out.

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


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

XLSX_PATH = Path(
    os.environ.get(
        "CLINICAL_NOTES_XLSX",
        "/home/ai/Workspace/Doha/MedVoice/src/data/real_use_cases.xlsx",
    )
)

PASS_THRESHOLD = float(os.environ.get("CLINICAL_NOTE_PASS_THRESHOLD", "0.65"))
JUDGE_MODEL = os.environ.get("CLINICAL_JUDGE_MODEL", "openai/gpt-6-luna")
# Bias mitigation: each scenario's judged fields are graded this many times,
# each with the field list presented in a freshly randomized order, and the
# per-field scores are averaged — see _judge_batch_multi for why.
JUDGE_RUNS = max(1, int(os.environ.get("CLINICAL_JUDGE_RUNS", "3")))

SKIP_SCENARIOS = {s.strip().upper() for s in os.environ.get("CLINICAL_NOTES_SKIP", "").split(",") if s.strip()}

# This workbook has no Mode/Specialty columns — every scenario is treated
# as narrated dictation, not a back-and-forth conversation. Flip a
# department's value here if that's wrong for it.
DEFAULT_IS_CONVERSATION = {"er": False, "opd": False, "dental": False}

SCENARIO_SHEETS = ["ER Scenarios", "OPD Scenarios", "Dental Scenarios"]
SCENARIO_ID_RE = re.compile(r"^([A-Za-z]+-\d+)\b")

# Extraction-JSON keys for the "always AI-recommended" fields that live
# outside model.department_fields.DEPARTMENT_FIELDS (see extract_features.py
# / department_fields.py docstrings for why they're excluded from there).
EXTRA_FIELDS = {
    "triage_priority": "Triage Priority",
    "icd10_codes": "ICD-10 Codes",
    "final_diagnosis": "Final Diagnosis",
    "final_diagnosis_rationale": "Final Diagnosis Rationale",
}

# Labels this workbook uses that don't literally match a department_fields
# label or an EXTRA_FIELDS label — map them to the same canonical label so
# they resolve to the same JSON key. Add to this as schema drift surfaces
# (see the SCHEMA-DRIFT WARNING printed at collection time).
LABEL_ALIASES = {
    "Triage Classification": "Triage Priority",
}

# Fields graded by ICD-10 code structure (parent/child match) instead of
# plain text similarity — see module docstring. Falls back to standard
# content grading when the Expected text has no code-shaped token in it,
# which is the common case in this workbook.
ICD10_FIELDS = {"icd10_codes", "final_diagnosis"}

# Fields intentionally left out of scoring entirely (not graded, not
# included in a scenario's percentage, not printed in the per-field
# report). "history_considered"/"history_notes" no longer apply since
# there is no History-ON pass in this workbook, but are kept here in case
# ExtractFeature still emits them so they don't pollute scoring.
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
    """Case/dash/punctuation/whitespace-insensitive label key, so e.g.
    'Chief Complaint\\xa0' (nbsp, in the sheet) matches 'Chief Complaint'
    (in code)."""
    return re.sub(r"[^a-z0-9]", "", (label or "").lower())


def _build_label_key_map(department: str) -> dict[str, str]:
    m = {}
    for f in get_department_fields(department):
        m[_norm_label(f["label"])] = f["key"]
    for key, label in EXTRA_FIELDS.items():
        m[_norm_label(label)] = key
    for alias_label, canonical_label in LABEL_ALIASES.items():
        canonical_key = m.get(_norm_label(canonical_label))
        if canonical_key is not None:
            m[_norm_label(alias_label)] = canonical_key
    return m


def _parse_scripts(wb) -> dict[str, dict]:
    """Scripts sheet is Departement | Language | Scenario, with no ID
    column and no Mode/Specialty/Patient Code columns. Scenario IDs are
    assigned by counting non-blank rows per department in order, giving
    "ER-1", "ER-2", "OPD-1", ... which is matched positionally against
    the scenario sheets' own title rows."""
    ws = wb["Scripts"]
    rows = list(ws.iter_rows(values_only=True))
    header, data_rows = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    counts: dict[str, int] = {}
    out = {}
    for row in data_rows:
        if not row or not row[idx["Departement"]]:
            continue
        dept_raw = str(row[idx["Departement"]]).strip()
        department = dept_raw.lower()
        language_raw = str(row[idx["Language"]]).strip().lower()
        script = row[idx["Scenario"]]
        counts[dept_raw] = counts.get(dept_raw, 0) + 1
        scenario_id = f"{dept_raw}-{counts[dept_raw]}"
        out[scenario_id] = {
            "department": department,
            "is_conversation": DEFAULT_IS_CONVERSATION.get(department, False),
            "language": "ar" if language_raw.startswith("ar") else "en",
            "specialty": None,
            "patient_code": None,
            "script": script,
        }
    return out


@dataclass
class FieldSpec:
    expected: str


def _parse_scenario_sheet(ws) -> dict[str, dict[str, FieldSpec]]:
    """Returns {scenario_id: {field_label: FieldSpec}}. Sub-tables here are
    just Field | Expected (2 columns) per scenario."""
    rows = list(ws.iter_rows(values_only=True))
    scenarios: dict[str, dict[str, FieldSpec]] = {}
    current_id: Optional[str] = None

    for row in rows:
        first = row[0] if len(row) > 0 else None
        second = row[1] if len(row) > 1 else None
        # Title row: e.g. "ER-1" with the rest of the row empty.
        if isinstance(first, str) and second is None:
            m = SCENARIO_ID_RE.match(first.strip())
            if m:
                current_id = m.group(1)
                scenarios.setdefault(current_id, {})
            continue
        if first == "Field" and second == "Expected":
            continue  # sub-table header row
        if current_id and first and second is not None:
            label = str(first).strip()
            expected = str(second).strip()
            scenarios[current_id][label] = FieldSpec(expected=expected)

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

    unmatched: dict[str, set[str]] = {}  # department -> set of unmatched labels

    scenarios = []
    for scenario_id, meta in scripts.items():
        if scenario_id in SKIP_SCENARIOS:
            continue
        if scenario_id not in scenario_fields:
            print(f"[collection] WARNING: {scenario_id} has a script but no matching "
                  f"Field/Expected sub-table in the scenario sheets — skipping it.")
            continue
        label_key_map = _build_label_key_map(meta["department"])
        expected_fields: dict[str, FieldSpec] = {}
        for label, spec in scenario_fields[scenario_id].items():
            key = label_key_map.get(_norm_label(label))
            if key is None:
                unmatched.setdefault(meta["department"], set()).add(label)
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

    if unmatched:
        print("\n" + "=" * 78)
        print("[collection] SCHEMA-DRIFT WARNING: these Expected-sheet labels did not "
              "match any department_fields/EXTRA_FIELDS key and will NOT be scored:")
        for dept, labels in unmatched.items():
            print(f"  {dept}: {sorted(labels)}")
        print("Add entries to LABEL_ALIASES (or fix the label/key on the pipeline side) "
              "to pick these up.")
        print("=" * 78 + "\n")

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

# Whole-string "nothing to report" phrasing — e.g. "Not specified.", "No
# neurological abnormality reported", "No significant abnormality
# reported." These are a clinician's way of writing "blank" in prose, so
# they're graded the same as an actually-blank field. Anchored to the
# WHOLE (period-stripped) string so a real negative finding that merely
# starts with "No ..." but carries extra content — e.g. "No known drug
# allergies", "Not applicable due to ICU admission." — is NOT swept in
# here; only phrases that are themselves just "nothing here" count.
_EMPTY_PHRASE_RE = re.compile(
    r"^(?:"
    r"not\s+(?:specified|applicable|mentioned|reported|noted|available|documented)"
    r"|no\b.*\b(?:reported|noted|specified|found|identified|documented|observed|detected|mentioned)"
    r")$",
    re.IGNORECASE,
)


def _is_none_or_blank_text(text: str) -> bool:
    """True for expected/actual text that IS (once any parenthetical
    aside and trailing period are stripped) nothing but 'none' / 'blank' /
    'n/a' / 'empty' / 'nil', possibly combined with '/' or 'or' — e.g.
    "None / blank" — OR a whole-string "nothing to report" phrase such as
    "Not specified." or "No neurological abnormality reported" (see
    _EMPTY_PHRASE_RE). Deliberately does NOT match real content that
    merely mentions one of those words in passing."""
    t = _PARENTHETICAL_RE.sub("", (text or "")).strip().rstrip(".").strip()
    if not t:
        return False
    tl = t.lower()
    if _EMPTY_PHRASE_RE.match(tl):
        return True
    parts = [p.strip() for p in _EMPTY_SPLIT_RE.split(tl) if p.strip()]
    return bool(parts) and all(_EMPTY_TOKEN_RE.match(p) for p in parts)


def _is_blank_like(text: str) -> bool:
    """True for an empty string OR a none/blank-equivalent value."""
    t = (text or "").strip()
    return (not t) or _is_none_or_blank_text(t)


def _field_bucket(expected_text: str) -> str:
    """No Source column in this workbook, so the bucket is determined
    purely from whether the Expected text itself is blank-equivalent."""
    return "empty" if _is_none_or_blank_text(expected_text) else "standard"


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
    "S61.409A"."""
    return code.upper()[:3]


def _grade_icd10_field(expected_text: str, actual_text: str) -> Optional[GradeResult]:
    """Deterministic parent/child ICD-10 grading. Returns None (caller
    should fall back to standard content grading) if the expected text
    doesn't contain anything code-shaped to compare against — which is
    the common case in this workbook, where these fields are usually
    plain diagnosis names."""
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


def _content_grade(key: str, expected_text: str, actual_val: str) -> Optional[GradeResult]:
    """Returns a GradeResult if the score is determinable without an LLM
    call (blank-equivalent fields, missing content, ICD-10 code
    comparisons), else None — meaning the caller should queue this field
    for the batch LLM judge."""
    bucket = _field_bucket(expected_text)
    if bucket == "empty":
        if _is_blank_like(actual_val):
            return GradeResult(1.0, "blank (or equivalent) as expected")
        return GradeResult(0.2, "populated a field that shouldn't be")
    if not actual_val:
        return GradeResult(0.0, "missing — expected content")
    if key in ICD10_FIELDS:
        icd_result = _grade_icd10_field(expected_text, actual_val)
        if icd_result is not None:
            return icd_result
    return None


_shared_judge_client = None


def _judge_client():
    global _shared_judge_client
    from openai import OpenAI

    api_key = getattr(Config, "OPENROUTER_API_KEY", None) or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("No OPENROUTER_API_KEY configured — cannot run the pipeline or the LLM judge.")
    if _shared_judge_client is None:
        _shared_judge_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _shared_judge_client


def _safe_parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


def _judge_batch(client, scenario_id: str, items: list[dict]) -> dict:
    """One LLM call scoring every field that needs semantic judgment for a
    scenario, instead of one call per field. Items are graded in the
    order given — callers doing repeated/shuffled sampling (see
    _judge_batch_multi) pass a pre-shuffled list."""
    if not items:
        return {}

    numbered = "\n\n".join(
        f'{i}. field_id: "{it["id"]}"\n   expected: "{it["expected"]}"\n   actual: "{it["actual"]}"'
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
the same fact is expressed.

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
        closest_idx = min(range(len(scores)), key=lambda i: abs(scores[i] - avg))
        verdict = per_field_verdicts[field_id][closest_idx]
        if spread > JUDGE_VARIANCE_FLAG_THRESHOLD:
            verdict = f"[judge variance {min(scores):.0f}-{max(scores):.0f}% across {runs} runs] {verdict}"
        merged[field_id] = {"score": avg, "verdict": verdict}
    return merged


# ═══════════════════════════════════════════════════════════════════════
# Per-scenario scoring (single pass — no History-ON in this workbook)
# ═══════════════════════════════════════════════════════════════════════

def score_scenario(scenario: Scenario, extracted_json: dict, client) -> list[FieldScore]:
    """Grades a scenario's single extraction pass against the workbook's
    Expected column. See module docstring for the full rubric."""
    results: list[FieldScore] = []
    to_judge: list[dict] = []  # {id, expected, actual} batch for the judge

    for key, spec in scenario.expected_fields.items():
        actual_val = _flatten(extracted_json.get(key))
        pre = _content_grade(key, spec.expected, actual_val)
        if pre is not None:
            results.append(FieldScore(key, spec.expected, actual_val, pre.score, pre.verdict))
            continue
        to_judge.append({"id": key, "expected": spec.expected, "actual": actual_val})

    judged = _judge_batch_multi(client, scenario.id, to_judge) if to_judge else {}
    for item in to_judge:
        key = item["id"]
        obj = judged.get(key, {})
        score = float(obj.get("score", 50)) / 100.0
        verdict = obj.get("verdict", "no judge response — defaulted to 50%")
        results.append(FieldScore(key, item["expected"], item["actual"], score, verdict))

    results.sort(key=lambda r: r.key)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def _print_scenario_report(scenario: Scenario, scores: list[FieldScore], pct: float) -> None:
    print(f"\n{'=' * 78}")
    header = (f"{scenario.id} ({scenario.department.upper()}, {scenario.language}, "
              f"{'conversation' if scenario.is_conversation else 'dictation'})  —  SCORE: {pct:.1%}")
    print(header)
    print(f"{'-' * 78}")
    for s in scores:
        flag = "OK  " if s.score >= 0.8 else ("~   " if s.score >= 0.5 else "FAIL")
        print(f"  [{flag}] {s.score:5.0%}  {s.key:<28} {s.verdict}")
        if s.score < 0.8:
            print(f"           expected: {s.expected[:90]}")
            print(f"           actual:   {s.actual[:90] or '(blank)'}")
    print(f"{'=' * 78}\n")


@pytest.fixture(scope="session")
def judge_client():
    return _judge_client()


@pytest.fixture(scope="session")
def session_results():
    results: list[tuple[str, float]] = []  # id, pct
    yield results
    if not results:
        return
    print(f"\n{'#' * 78}")
    print("CLINICAL NOTE FIELD EXTRACTION — SUMMARY")
    print(f"{'#' * 78}")
    for scenario_id, pct in results:
        bar = "#" * int(pct * 40)
        print(f"  {scenario_id:<10} SCORE {pct:5.1%}  {bar}")
    avg = sum(p for _, p in results) / len(results)
    print(f"{'-' * 78}")
    print(f"  {'AVERAGE':<10} {avg:5.1%}")
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

    extracted_json, _ = ExtractFeature.extract(
        translated, department=scenario.department, patient_history_text="", is_conversation=scenario.is_conversation
    )
    extracted_json = extracted_json or {}

    scores = score_scenario(scenario, extracted_json, judge_client)
    pct = sum(s.score for s in scores) / len(scores) if scores else 0.0

    session_results.append((scenario.id, pct))
    _print_scenario_report(scenario, scores, pct)

    failing = [s for s in scores if s.score < 0.5]
    detail = "\n".join(f"  - {s.key}: {s.score:.0%} ({s.verdict})" for s in failing)
    assert pct >= PASS_THRESHOLD, (
        f"{scenario.id} scored {pct:.1%}, below threshold {PASS_THRESHOLD:.0%}.\n"
        f"Worst fields:\n{detail}"
    )