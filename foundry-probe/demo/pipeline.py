"""
demo/pipeline.py – Structuring, extraction, reasoning pipeline steps.
Python 3.9 compatible.
SECURITY: never logs transcripts, prompts with patient data, or keys.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
import rapidfuzz.fuzz as rfuzz

from config import client, budget_manager
from demo.model_profiles import build_kwargs

logger = logging.getLogger(__name__)

# ── Term dictionary ───────────────────────────────────────────────────────────
# 60 common drugs/terms (bilingual, Arabic→English)
TERMS: Dict[str, str] = {
    # Drugs – Arabic
    "باراسيتامول": "paracetamol", "أسبرين": "aspirin", "إيبوبروفين": "ibuprofen",
    "أموكسيسيلين": "amoxicillin", "أزيثروميسين": "azithromycin",
    "ميتفورمين": "metformin", "أتورفاستاتين": "atorvastatin",
    "ليسينوبريل": "lisinopril", "أملوديبين": "amlodipine",
    "لوسارتان": "losartan", "إنالابريل": "enalapril",
    "فالسارتان": "valsartan", "ميترونيدازول": "metronidazole",
    "سيبروفلوكساسين": "ciprofloxacin", "كلاريثروميسين": "clarithromycin",
    "أوميبرازول": "omeprazole", "بانتوبرازول": "pantoprazole",
    "لانسوبرازول": "lansoprazole", "رانيتيدين": "ranitidine",
    "سالبوتامول": "salbutamol", "بيتاميثازون": "betamethasone",
    "ديكساميثازون": "dexamethasone", "بريدنيزون": "prednisone",
    "وارفارين": "warfarin", "هيبارين": "heparin",
    "إنسولين": "insulin", "غليبنكلاميد": "glibenclamide",
    "ديجوكسين": "digoxin", "فيروسيميد": "furosemide",
    "سبيرونولاكتون": "spironolactone", "هيدروكلوروثيازيد": "hydrochlorothiazide",
    # Drugs – English (normalisation)
    "paracetamol": "paracetamol", "acetaminophen": "paracetamol",
    "aspirin": "aspirin", "ibuprofen": "ibuprofen",
    "amoxicillin": "amoxicillin", "azithromycin": "azithromycin",
    "metformin": "metformin", "atorvastatin": "atorvastatin",
    "lisinopril": "lisinopril", "amlodipine": "amlodipine",
    "losartan": "losartan", "enalapril": "enalapril",
    "valsartan": "valsartan", "metronidazole": "metronidazole",
    "ciprofloxacin": "ciprofloxacin", "clarithromycin": "clarithromycin",
    "omeprazole": "omeprazole", "pantoprazole": "pantoprazole",
    "salbutamol": "salbutamol", "albuterol": "salbutamol",
    "dexamethasone": "dexamethasone", "prednisone": "prednisone",
    "warfarin": "warfarin", "heparin": "heparin",
    "insulin": "insulin", "digoxin": "digoxin",
    "furosemide": "furosemide", "spironolactone": "spironolactone",
}

RED_FLAGS: List[str] = [
    "chest pain", "ألم في الصدر", "وجع في الصدر",
    "stroke signs", "علامات السكتة",
    "sepsis", "إنتان",
    "anaphylaxis", "حساسية شديدة",
    "suicidal ideation", "أفكار انتحارية",
    "loss of consciousness", "فقدان الوعي",
    "severe bleeding", "نزيف شديد",
    "difficulty breathing", "ضيق تنفس",
]

# Pairs that are dangerously similar – never auto-correct between them
AMBIGUOUS_PAIRS: List[Tuple[str, str]] = [
    ("hydralazine", "hydroxyzine"),
    ("cefazolin", "cefoxitin"),
    ("prednisone", "prednisolone"),
    ("lorazepam", "clonazepam"),
]


def _is_ambiguous(original: str, replacement: str) -> bool:
    orig_lower = original.lower()
    repl_lower = replacement.lower()
    for a, b in AMBIGUOUS_PAIRS:
        if (orig_lower in (a, b)) and (repl_lower in (a, b)) and orig_lower != repl_lower:
            return True
    return False


def term_correction(transcript: str) -> Tuple[str, List[Dict[str, Any]], float]:
    t0 = time.perf_counter()
    words = transcript.split()
    corrections: List[Dict[str, Any]] = []

    for i, word in enumerate(words):
        best_match: Optional[str] = None
        best_score: float = 0.0

        for term, replacement in TERMS.items():
            score = rfuzz.ratio(word.lower(), term.lower())
            if score > 88:
                if best_match is not None and best_match != replacement:
                    best_match = None  # Ambiguous – skip
                    break
                best_match = replacement
                best_score = score

        if (
            best_match
            and best_match.lower() != word.lower()
            and not _is_ambiguous(word, best_match)
        ):
            words[i] = best_match
            corrections.append({
                "original": word,
                "corrected": best_match,
                "score": round(best_score, 1),
            })

    return " ".join(words), corrections, time.perf_counter() - t0


# ── Structuring ───────────────────────────────────────────────────────────────
_STRUCT_SYSTEM = """\
You are a medical transcription structurer.
Rules:
1. Translate Arabic or mixed Arabic/English text to English.
2. Organise into exactly these sections (omit empty ones):
   ## Complaint and History
   ## Examination
   ## Assessment
   ## Plan
3. NEVER add, infer, or sharpen any clinical detail.
4. Keep negation, laterality, doses, durations EXACTLY as stated.
5. Mark unclear speech as [unclear], never guess.
6. After the note, emit a JSON block inside <signals>…</signals> with:
{
  "problem_count": <int>,
  "comorbidity_count": <int>,
  "medication_count": <int>,
  "special_population": "pregnancy|paediatric|elderly|none",
  "er_abnormal_primary_survey": <bool>,
  "red_flags": [],
  "sentence_source_links": {}
}
"""

_SIGNALS_RE = re.compile(r"<signals>(.*?)</signals>", re.DOTALL)


async def run_structuring(
    transcript: str,
    deployment: str,
) -> Tuple[str, Dict[str, Any], float]:
    t0 = time.perf_counter()
    kwargs = build_kwargs(deployment, max_tokens=1200, temperature=0.0)
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _STRUCT_SYSTEM},
            {"role": "user",   "content": transcript},
        ],
        **kwargs,
    )
    latency = time.perf_counter() - t0

    if response.usage:
        budget_manager.add_cost(
            deployment,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    content = response.choices[0].message.content or ""
    m = _SIGNALS_RE.search(content)
    signals: Dict[str, Any] = {}
    structured = content
    if m:
        try:
            signals = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
        structured = content.replace(m.group(0), "").strip()

    return structured, signals, latency


# ── Extraction ────────────────────────────────────────────────────────────────
_DEPT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "ER": {
        "type": "object",
        "properties": {
            "triage_category":    {"type": ["string", "null"], "enum": ["Red", "Yellow", "Green", "Unknown", None]},
            "mechanism_of_injury":{"type": ["string", "null"]},
            "consent":            {"type": ["string", "null"], "enum": ["Yes", "No", "Verbal", "Not Applicable", None]},
        },
        "required": ["triage_category", "mechanism_of_injury", "consent"],
        "additionalProperties": False,
    },
    "OPD": {
        "type": "object",
        "properties": {
            "follow_up_weeks": {"type": ["integer", "null"]},
            "referral_needed": {"type": ["boolean", "null"]},
            "vitals_checked":  {"type": ["boolean", "null"]},
        },
        "required": ["follow_up_weeks", "referral_needed", "vitals_checked"],
        "additionalProperties": False,
    },
    "Dental": {
        "type": "object",
        "properties": {
            "tooth_number":    {"type": ["string", "null"]},
            "procedure":       {"type": ["string", "null"], "enum": ["Extraction", "Filling", "Cleaning", "Root Canal", "Other", None]},
            "anesthesia_used": {"type": ["boolean", "null"]},
        },
        "required": ["tooth_number", "procedure", "anesthesia_used"],
        "additionalProperties": False,
    },
}

_EXTRACT_SYSTEM = (
    "Extract fields for the {dept} department into the JSON schema. "
    "If a value is not explicitly mentioned, output null. "
    "No diagnoses. No ICD codes. Never infer."
)


async def _repair_json(
    deployment: str, bad_text: str, schema: Dict[str, Any]
) -> Dict[str, Any]:
    """One retry asking model to fix malformed JSON."""
    kwargs = build_kwargs(deployment, max_tokens=400, temperature=0.0,
                          response_format={"type": "json_object"})
    resp = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Fix the following text so it is valid JSON matching this schema: "
             + json.dumps(schema)},
            {"role": "user", "content": bad_text},
        ],
        **kwargs,
    )
    if resp.usage:
        budget_manager.add_cost(deployment, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return json.loads(resp.choices[0].message.content or "{}")


async def run_extraction(
    structured_text: str,
    department: str,
    deployment: str,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    dept = department if department in _DEPT_SCHEMAS else "OPD"
    schema = _DEPT_SCHEMAS[dept]

    rf = {"type": "json_schema", "json_schema": {"name": f"{dept.lower()}_extraction",
          "strict": True, "schema": schema}}
    kwargs = build_kwargs(deployment, max_tokens=400, temperature=0.0,
                          response_format=rf)

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM.format(dept=dept)},
            {"role": "user",   "content": structured_text},
        ],
        **kwargs,
    )
    latency = time.perf_counter() - t0
    if response.usage:
        budget_manager.add_cost(deployment, response.usage.prompt_tokens, response.usage.completion_tokens)

    raw = response.choices[0].message.content or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Extraction JSON parse failed for %s, attempting repair", deployment)
        result = await _repair_json(deployment, raw, schema)

    return result, latency


# ── Reasoning ─────────────────────────────────────────────────────────────────
_REASON_SYSTEM = """\
You are a clinical reasoning assistant. Return ONLY a valid JSON object — no markdown, no backticks.
Use ICD-10-AM (the Australian/KSA edition). Schema:
{
  "icd10": [{"code": "A00.0", "description": "...", "evidence_sentences": ["..."]}],
  "final_diagnosis": "...",
  "rationale": "...",
  "er_triage_priority": "Immediate|Urgent|Semi-urgent|Non-urgent|null",
  "clinical_gap_questions": ["..."],
  "plan_review": "..."
}
"""


async def run_reasoning(
    structured_text: str,
    history_text: str,
    deployment: str,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    content = structured_text
    if history_text and history_text.strip():
        content += (
            "\n\nPATIENT HISTORY (use ONLY to inform icd10/final_diagnosis/rationale):\n"
            + history_text
        )

    kwargs = build_kwargs(deployment, max_tokens=800, temperature=0.0,
                          response_format={"type": "json_object"})

    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _REASON_SYSTEM},
            {"role": "user",   "content": content},
        ],
        **kwargs,
    )
    latency = time.perf_counter() - t0
    if response.usage:
        budget_manager.add_cost(deployment, response.usage.prompt_tokens, response.usage.completion_tokens)

    raw = response.choices[0].message.content or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Reasoning JSON parse failed for %s, attempting repair", deployment)
        result = await _repair_json(deployment, raw, {})

    return result, latency
