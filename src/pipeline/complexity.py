"""Deterministic 5-Rule Complexity Classification Engine.

Implements high-performance, deterministic case complexity classification
to replace the LLM model-router call with sub-millisecond execution.

Routing Rule:
A case is classified as 'complex' (Tier: gpt-5.6-sol) if and only if
>= 3 of the following 5 clinical criteria are satisfied:
1. Noisy dictation: >= 2 speech fillers or self-corrections.
2. Multi-speaker dispersion: Companion present or >= 10 dialogue turns.
3. Reasoning cues: Diagnostic uncertainty, differential, or conditional plan.
4. History relevant: History active AND past ICD matches current complaint chapter.
5. Unknown abbreviations: >= 1 uppercase medical acronym not in verified whitelist.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

WHITELIST_PATH = Path(__file__).resolve().parents[1] / "data" / "abbreviation_whitelist.json"

# ── Compiled Patterns & Glossaries ───────────────────────────────────────────

# Rule 1: Speech fillers & self-corrections (English + Arabic)
# e.g.: okay, sorry, i mean, mean to say, no wait, عفوا, اقصد, يعني, قصدي
_FILLER_PATTERNS = [
    re.compile(r"\b(?:okay|ok|sorry|i mean|mean to say|no wait|wait no|scratch that)\b", re.IGNORECASE),
    re.compile(r"(?:\b|\s|^)(?:عفوا|اقصد|يعني|قصدي|لا استني|لا انتظر)(?:\b|\s|$)", re.UNICODE),
]

# Rule 2: Multi-speaker / companion keywords & turn pattern
_COMPANION_PATTERNS = [
    re.compile(r"\b(?:daughter|wife|husband|mother|father|son|sister|brother|companion|escort|family member|relative|carer|caregiver)\b", re.IGNORECASE),
    re.compile(r"(?:\b|\s|^)(?:الابنه|ابنه|الزوجه|زوجته|زوجه|زوجها|الام|والدته|الاب|والده|المرافق|مرافق|مرافقه|المرافقه|اهل المريض)(?:\b|\s|$)", re.UNICODE),
]
_SPEAKER_TURN_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:(?:doctor|physician|dr|patient|pt|daughter|wife|companion|relative|speaker\s*\d+|person\s*\d+|طبيب|دكتور|المريض|مريض|المرافق)\s*[:：]|-|\*)",
    re.IGNORECASE | re.MULTILINE,
)

# Rule 3: Reasoning / uncertainty cues & conditional plans
_REASONING_PATTERNS = [
    re.compile(r"\b(?:possibly|probable|probably|could be|rule out|r/o|differential|provisional|borderline|equivocal|suspected|suspect|vs|versus)\b", re.IGNORECASE),
    re.compile(r"\bif\b.+?\b(?:otherwise|else)\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:\b|\s|^)(?:نستبعد|استبعاد|احتمال|محتمل|ممكن|مشتبه|شك في|تشخيص تفريقي|مبدئي|تفاضلي|غير موكد|حاله اشتباه)(?:\b|\s|$)", re.UNICODE),
    re.compile(r"(?:لو|اذا).+?(?:غير كده|والا)", re.UNICODE | re.DOTALL),
]

# Rule 5: Candidate medical acronyms (2 to 6 uppercase letters/digits)
_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")

# Words to ignore from acronym detection (common English words / units / modes)
_ACRONYM_IGNORE = frozenset({
    "THE", "AND", "FOR", "NOT", "YES", "NO", "WITH", "FROM", "OUT",
    "CAN", "MAY", "HAD", "HAS", "WAS", "ARE", "DAY", "DAYS", "MIN", "SEC",
    "NEW", "OLD", "NOW", "ALL", "BUT", "OFF", "ONE", "TWO", "PER",
    "DR", "MR", "MRS", "MS", "PT", "AGE", "SEX", "YEAR", "MALE", "LEFT", "RIGHT",
})


def _load_abbreviation_whitelist() -> Set[str]:
    """Load verified clinical abbreviation whitelist from JSON file."""
    try:
        if WHITELIST_PATH.exists():
            with open(WHITELIST_PATH, encoding="utf-8") as f:
                data = json.load(f)
                return set(item.upper().strip() for item in data.get("whitelist", []))
    except Exception as exc:
        logger.warning("[ComplexityEngine] Failed to load abbreviation whitelist: %s", exc)
    return set()


_ABBREVIATION_WHITELIST = _load_abbreviation_whitelist()


# ── Text Normalization ───────────────────────────────────────────────────────

def normalize_text_for_rules(text: str) -> str:
    """Normalize text (NFKC, casefold, unify Arabic alefs/yahs)."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    # Unify Arabic variants
    norm = norm.translate(str.maketrans({
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
    }))
    return norm


# ── ICD-10 Chapter Code Mapping ──────────────────────────────────────────────

def icd10_chapter(code: str) -> str:
    """Return the ICD-10 chapter prefix/category for broad clinical domain match."""
    if not code:
        return ""
    code_clean = code.strip().upper().split(" ")[0].split(".")[0]
    if not code_clean:
        return ""
    letter = code_clean[0]
    try:
        num = int(code_clean[1:3]) if len(code_clean) >= 3 and code_clean[1:3].isdigit() else 0
    except ValueError:
        num = 0

    # Official WHO ICD-10 Chapters
    if letter in ("A", "B"):
        return "I_Infectious"
    if letter == "C" or (letter == "D" and num <= 49):
        return "II_Neoplasms"
    if letter == "D" and num >= 50:
        return "III_Blood"
    if letter == "E":
        return "IV_Endocrine"
    if letter == "F":
        return "V_Mental"
    if letter == "G":
        return "VI_Nervous"
    if letter == "H" and num <= 59:
        return "VII_Eye"
    if letter == "H" and num >= 60:
        return "VIII_Ear"
    if letter == "I":
        return "IX_Circulatory"
    if letter == "J":
        return "X_Respiratory"
    if letter == "K":
        return "XI_Digestive"
    if letter == "L":
        return "XII_Skin"
    if letter == "M":
        return "XIII_Musculoskeletal"
    if letter == "N":
        return "XIV_Genitourinary"
    if letter == "O":
        return "XV_Pregnancy"
    if letter == "P":
        return "XVI_Perinatal"
    if letter == "Q":
        return "XVII_Congenital"
    if letter == "R":
        return "XVIII_Symptoms_NotElsewhereClassified"
    if letter in ("S", "T"):
        return "XIX_Injury"
    if letter in ("V", "W", "X", "Y"):
        return "XX_External_Causes"
    if letter == "Z":
        return "XXI_Health_Services"
    if letter == "U":
        return "XXII_Special_Purposes"
    return letter


# ── Individual Rule Evaluators ───────────────────────────────────────────────

def evaluate_noisy_dictation(text: str) -> Tuple[bool, List[str]]:
    """Rule 1: >= 2 disfluencies, fillers, or self-corrections."""
    norm = normalize_text_for_rules(text)
    matches: List[str] = []
    for pattern in _FILLER_PATTERNS:
        for m in pattern.finditer(norm):
            matches.append(m.group(0).strip())
    # Count total occurrences
    passed = len(matches) >= 2
    return passed, matches


def evaluate_multi_speaker(text: str, is_conversation: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Rule 2: Companion present OR >= 10 dialogue turns."""
    norm = normalize_text_for_rules(text)
    companion_found: List[str] = []
    for pattern in _COMPANION_PATTERNS:
        for m in pattern.finditer(norm):
            companion_found.append(m.group(0).strip())

    turns = len(_SPEAKER_TURN_PATTERN.findall(text))
    # If mode is explicitly conversation or turns are formatted
    turn_condition = turns >= 10 or (is_conversation and (turns >= 6 or len(text.splitlines()) >= 10))
    passed = bool(companion_found) or turn_condition
    return passed, {
        "companion_matches": companion_found,
        "turn_count": turns,
        "is_conversation": is_conversation,
    }


def evaluate_reasoning_cues(text: str) -> Tuple[bool, List[str]]:
    """Rule 3: Reasoning / uncertainty cues or conditional plans."""
    norm = normalize_text_for_rules(text)
    cues_found: List[str] = []
    for pattern in _REASONING_PATTERNS:
        for m in pattern.finditer(norm):
            matched = m.group(0).strip()
            # truncate long conditional matches
            cues_found.append(matched[:50] + ("..." if len(matched) > 50 else ""))
    passed = len(cues_found) >= 1
    return passed, cues_found


def evaluate_history_relevance(
    use_history: bool,
    patient_history_visits: Sequence[Any],
    current_icd_candidates: Sequence[Tuple[str, str]],
) -> Tuple[bool, Dict[str, Any]]:
    """Rule 4: History ON and past visit has an ICD code in the same chapter as current complaint."""
    if not use_history or not patient_history_visits:
        return False, {"reason": "history_off_or_empty"}

    # Collect past ICD chapters
    history_chapters: Set[str] = set()
    history_codes: List[str] = []
    for v in patient_history_visits:
        diagnoses = []
        if isinstance(v, dict):
            diagnoses = v.get("icd10_diagnoses") or []
        elif hasattr(v, "icd10_diagnoses"):
            diagnoses = getattr(v, "icd10_diagnoses") or []
        for d in diagnoses:
            if isinstance(d, str) and d.strip():
                history_codes.append(d.strip())
                ch = icd10_chapter(d)
                if ch:
                    history_chapters.add(ch)

    # Collect current ICD candidate chapters
    candidate_chapters: Set[str] = set()
    for item in (current_icd_candidates or []):
        code = item[0] if isinstance(item, (tuple, list)) and item else str(item)
        ch = icd10_chapter(code)
        if ch:
            candidate_chapters.add(ch)

    overlap = history_chapters.intersection(candidate_chapters)
    passed = len(overlap) > 0
    return passed, {
        "history_chapters": list(history_chapters),
        "candidate_chapters": list(candidate_chapters),
        "overlapping_chapters": list(overlap),
    }


def evaluate_unknown_abbreviations(text: str) -> Tuple[bool, List[str]]:
    """Rule 5: >= 1 uppercase medical acronym not found in verified whitelist."""
    # Find all uppercase word tokens
    raw_acronyms = set(_ACRONYM_PATTERN.findall(text))
    unknown: List[str] = []
    whitelist = _ABBREVIATION_WHITELIST

    for acr in sorted(raw_acronyms):
        acr_upper = acr.upper().strip()
        if acr_upper in _ACRONYM_IGNORE:
            continue
        # If it contains digits (like FDI tooth 36 or stage T2), ignore if not pure acronym
        if acr_upper.isdigit():
            continue
        if acr_upper not in whitelist:
            unknown.append(acr)

    passed = len(unknown) >= 1
    return passed, unknown


# ── Complexity Evaluation Data Model ────────────────────────────────────────

@dataclass
class ComplexityEvaluation:
    """Outcome of the deterministic 5-rule complexity classifier."""
    score: int
    is_complex: bool
    tier: str  # "complex" | "simple"
    matched_rules: List[str] = field(default_factory=list)
    rule_details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "is_complex": self.is_complex,
            "tier": self.tier,
            "matched_rules": list(self.matched_rules),
            "rule_details": self.rule_details,
            "latency_ms": self.latency_ms,
        }


# ── Main Classification Entry Point ─────────────────────────────────────────

def classify_clinical_complexity(
    text: str,
    is_conversation: bool = False,
    use_history: bool = False,
    patient_history_visits: Sequence[Any] = (),
    current_icd_candidates: Sequence[Tuple[str, str]] = (),
) -> ComplexityEvaluation:
    """Evaluate all 5 clinical rules and classify case complexity in < 1 millisecond.

    Threshold:
    score >= 3 -> complex tier (served by gpt-5.6-sol)
    score < 3  -> simple tier (served by gpt-5.6-luna)
    """
    t0 = time.perf_counter()
    matched_rules: List[str] = []
    rule_details: Dict[str, Any] = {}

    # 1. Noisy dictation
    r1_pass, r1_evidence = evaluate_noisy_dictation(text)
    rule_details["noisy_dictation"] = {"passed": r1_pass, "matches": r1_evidence}
    if r1_pass:
        matched_rules.append("NOISY_DICTATION")

    # 2. Multi-speaker dispersion
    r2_pass, r2_evidence = evaluate_multi_speaker(text, is_conversation=is_conversation)
    rule_details["multi_speaker"] = {"passed": r2_pass, **r2_evidence}
    if r2_pass:
        matched_rules.append("MULTI_SPEAKER")

    # 3. Reasoning cues
    r3_pass, r3_evidence = evaluate_reasoning_cues(text)
    rule_details["reasoning_cues"] = {"passed": r3_pass, "cues": r3_evidence}
    if r3_pass:
        matched_rules.append("REASONING_CUES")

    # 4. History relevance
    r4_pass, r4_evidence = evaluate_history_relevance(
        use_history=use_history,
        patient_history_visits=patient_history_visits,
        current_icd_candidates=current_icd_candidates,
    )
    rule_details["history_relevant"] = {"passed": r4_pass, **r4_evidence}
    if r4_pass:
        matched_rules.append("HISTORY_RELEVANT")

    # 5. Unknown abbreviations
    r5_pass, r5_evidence = evaluate_unknown_abbreviations(text)
    rule_details["unknown_abbreviations"] = {"passed": r5_pass, "unknown": r5_evidence}
    if r5_pass:
        matched_rules.append("UNKNOWN_ABBREVIATIONS")

    score = len(matched_rules)
    is_complex = score >= 3
    tier = "complex" if is_complex else "simple"
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return ComplexityEvaluation(
        score=score,
        is_complex=is_complex,
        tier=tier,
        matched_rules=matched_rules,
        rule_details=rule_details,
        latency_ms=latency_ms,
    )
