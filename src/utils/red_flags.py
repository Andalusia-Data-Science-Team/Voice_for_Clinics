"""Lexicon-based candidate red-flag detection for clinician review."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

LEXICON_PATH = Path(__file__).resolve().parents[1] / "data" / "red_flag_lexicon.json"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(
        char for char in text
        if not unicodedata.category(char).startswith("M") and char != "\u0640"
    )
    text = text.translate(str.maketrans({
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
    }))
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def _load_lexicon(path: Path = LEXICON_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        lexicon = json.load(file)
    if not isinstance(lexicon, list):
        raise ValueError(f"Red-flag lexicon must be a JSON list: {path}")

    for entry in lexicon:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not isinstance(entry.get("label"), str)
            or not isinstance(entry.get("phrases"), dict)
        ):
            raise ValueError(f"Invalid red-flag lexicon entry in {path}: {entry!r}")
        for language, phrases in entry["phrases"].items():
            if language not in {"en", "ar"} or not isinstance(phrases, list):
                raise ValueError(
                    "Invalid phrase list in red-flag lexicon entry: "
                    f"{entry!r}"
                )
            if any(
                not isinstance(phrase, str) or not phrase.strip()
                for phrase in phrases
            ):
                raise ValueError(
                    f"Empty or invalid phrase in red-flag lexicon entry: {entry!r}"
                )
    return lexicon


_LEXICON = _load_lexicon()
_NORMALIZED_LEXICON = [
    (
        entry["id"],
        entry["label"],
        language,
        phrase,
        _normalize(phrase),
    )
    for entry in _LEXICON
    for language, phrases in entry["phrases"].items()
    for phrase in phrases
]

_NEGATION_CUES = {
    "en": (
        "denies", "deny", "denied", "no", "not", "without", "negative for",
        "absent", "never", "does not have", "do not have", "has no",
        "not experiencing",
    ),
    "ar": (
        "لا", "ليس", "ليست", "لم", "لن", "بدون", "ينفي", "تنفي",
        "لا يعاني من", "لا تعاني من", "لا يوجد", "لا توجد",
    ),
}
_UNCERTAINTY_CUES = {
    "en": (
        "possible", "possibly", "maybe", "perhaps", "might", "may",
        "could", "suspected", "uncertain", "rule out", "not sure",
    ),
    "ar": ("محتمل", "محتملة", "ربما", "قد", "اشتباه", "غير مؤكد"),
}
_CONTRAST_CUES = {"but", "however", "although", "except", "لكن", "ولكن"}
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?؟;؛\n]+")


def _has_recent_cue(
    preceding_words: list[str],
    cues: tuple[str, ...],
) -> bool:
    """Check whether a cue occurs close to a finding in the same clause."""
    if not preceding_words:
        return False
    last_contrast = max(
        (index for index, word in enumerate(preceding_words) if word in _CONTRAST_CUES),
        default=-1,
    )
    context = preceding_words[last_contrast + 1:]
    cue_sequences = [cue.split() for cue in cues]
    for cue_words in cue_sequences:
        width = len(cue_words)
        for start in range(max(0, len(context) - 6), len(context) - width + 1):
            if context[start:start + width] == cue_words:
                return True
    return False


def detect_red_flags(text: str) -> dict[str, Any]:
    """Return phrase matches with conservative, local negation classification."""
    matches = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _SENTENCE_BOUNDARY_RE.split(text):
        words = _normalize(sentence).split()
        if not words:
            continue
        for (
            flag_id,
            label,
            language,
            original_phrase,
            normalized_phrase,
        ) in _NORMALIZED_LEXICON:
            phrase_words = normalized_phrase.split()
            width = len(phrase_words)
            for start in range(len(words) - width + 1):
                if words[start:start + width] != phrase_words:
                    continue
                preceding = words[:start]
                following = words[start + width:]
                negated = _has_recent_cue(
                    preceding,
                    _NEGATION_CUES[language],
                )
                negated = negated or _has_recent_cue(
                    following[:3],
                    ("denied", "absent", "not present", "منفي", "غير موجود"),
                )
                uncertain = _has_recent_cue(preceding, _UNCERTAINTY_CUES[language])
                if uncertain:
                    status = "uncertain"
                elif negated:
                    status = "negated"
                else:
                    status = "affirmed"
                key = (flag_id, normalized_phrase, status)
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "id": flag_id,
                    "label": label,
                    "matched_phrase": original_phrase,
                    "language": language,
                    "status": status,
                })

    requires_review = any(item["status"] != "negated" for item in matches)
    if requires_review:
        action = "clinician_review"
        notice = (
            "Review the source and context. Negation handling is a simple "
            "phrase check and may be wrong; it is not a diagnosis or triage "
            "decision."
        )
    elif matches:
        action = "negated_only"
        notice = (
            "Listed warning phrase(s) appear negated in the text. Confirm "
            "context; this phrase check is not a clinical assessment."
        )
    else:
        action = "none_detected"
        notice = (
            "No lexicon phrase matched. This does not rule out an urgent "
            "condition."
        )

    return {
        "requires_clinician_review": requires_review,
        "action": action,
        "matched_signals": matches,
        "notice": notice,
    }
