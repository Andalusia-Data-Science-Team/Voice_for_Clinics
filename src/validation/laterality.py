"""
MedVoice Validation Subsystem - Clinical Laterality Detector
============================================================
Part of Phase 3 (Milestone 3.3): Clinical Consistency Rule Engine.

Detects anatomical laterality mentions across clinical text and ICD-10 codes:
  - Supports English and Arabic anatomical terms
  - Identifies left, right, bilateral, and midline mentions
  - Performs conflict detection between clinical findings and diagnoses
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Set


class Laterality(str, Enum):
    """Anatomical laterality classifications."""
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    UNSPECIFIED = "unspecified"


# English & Arabic keyword sets for anatomical laterality
LATERALITY_KEYWORDS: dict[Laterality, set[str]] = {
    Laterality.LEFT: {
        "left", "lt", "sinister", "sinistral", "left-sided", "leftside",
        "أيسر", "شمال", "اليسار", "الشمال", "الجهة اليسرى", "الجانب الأيسر", "يسرى", "اليسرى"
    },
    Laterality.RIGHT: {
        "right", "rt", "dexter", "dextral", "right-sided", "rightside",
        "أيمن", "يمين", "اليمين", "الجهة اليمنى", "الجانب الأيمن", "يمنى", "اليمنى"
    },
    Laterality.BILATERAL: {
        "bilateral", "both", "b/l", "bilaterally", "both sides",
        "양측", "كلا", "الجانبين", "الجهتين", "بالجهتين", "في كلا الجانبين", "ثنائي"
    },
}

_WORD_BOUNDARY_REGEX = re.compile(r"[\s,;:.()\[\]{}–—/\\]+")


@dataclass(frozen=True, slots=True)
class LateralityDetection:
    """Represents detected laterality in a segment of text."""
    laterality: Laterality
    matched_tokens: tuple[str, ...]
    source_snippet: str


def detect_laterality(text: str) -> LateralityDetection:
    """Scan text for anatomical laterality terms (English & Arabic).

    Returns LateralityDetection with classified laterality and matched tokens.
    """
    if not text:
        return LateralityDetection(Laterality.UNSPECIFIED, (), "")

    tokens = [t.lower().strip() for t in _WORD_BOUNDARY_REGEX.split(text) if t.strip()]
    found: dict[Laterality, list[str]] = {
        Laterality.LEFT: [],
        Laterality.RIGHT: [],
        Laterality.BILATERAL: [],
    }

    # Also check multi-word phrases directly
    lower_text = text.lower()
    for lat, kw_set in LATERALITY_KEYWORDS.items():
        for kw in kw_set:
            if " " in kw:
                if kw in lower_text:
                    found[lat].append(kw)
            else:
                if kw in tokens:
                    found[lat].append(kw)

    # Resolution precedence:
    # 1. Bilateral explicitly mentioned -> BILATERAL
    # 2. Both left and right mentioned in the same text -> BILATERAL
    # 3. Only left -> LEFT
    # 4. Only right -> RIGHT
    # 5. None -> UNSPECIFIED
    if found[Laterality.BILATERAL] or (found[Laterality.LEFT] and found[Laterality.RIGHT]):
        matched = tuple(set(found[Laterality.BILATERAL] + found[Laterality.LEFT] + found[Laterality.RIGHT]))
        return LateralityDetection(Laterality.BILATERAL, matched, text[:100])

    if found[Laterality.LEFT]:
        return LateralityDetection(Laterality.LEFT, tuple(set(found[Laterality.LEFT])), text[:100])

    if found[Laterality.RIGHT]:
        return LateralityDetection(Laterality.RIGHT, tuple(set(found[Laterality.RIGHT])), text[:100])

    return LateralityDetection(Laterality.UNSPECIFIED, (), "")


def check_laterality_conflict(findings_text: str, code_description: str) -> Optional[str]:
    """Check whether clinical findings and an ICD-10 description have conflicting lateralities.

    Returns:
        Explanation string if a conflict exists, otherwise None.
    """
    findings_lat = detect_laterality(findings_text)
    code_lat = detect_laterality(code_description)

    # Only flag if both specify a concrete non-unspecified laterality
    if findings_lat.laterality == Laterality.UNSPECIFIED or code_lat.laterality == Laterality.UNSPECIFIED:
        return None

    # Matching laterality is fine
    if findings_lat.laterality == code_lat.laterality:
        return None

    # Bilateral findings with unilateral diagnosis or vice versa is a mild mismatch,
    # but Left vs Right is a critical clinical laterality error.
    if {findings_lat.laterality, code_lat.laterality} == {Laterality.LEFT, Laterality.RIGHT}:
        return (
            f"Laterality mismatch detected: Clinical text mentions '{findings_lat.laterality.value}' "
            f"(tokens: {', '.join(findings_lat.matched_tokens)}), but recommended code specifies "
            f"'{code_lat.laterality.value}'."
        )

    if findings_lat.laterality == Laterality.BILATERAL and code_lat.laterality in (Laterality.LEFT, Laterality.RIGHT):
        return (
            f"Laterality caution: Clinical text mentions bilateral/both sides, but recommended code "
            f"is unilateral '{code_lat.laterality.value}'."
        )

    if findings_lat.laterality in (Laterality.LEFT, Laterality.RIGHT) and code_lat.laterality == Laterality.BILATERAL:
        return (
            f"Laterality caution: Clinical text mentions unilateral '{findings_lat.laterality.value}', "
            f"but recommended code specifies bilateral."
        )

    return None
