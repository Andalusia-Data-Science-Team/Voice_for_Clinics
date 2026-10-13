"""
src/schemas/input_schemas.py
============================
Strict Pydantic v2 input schemas for every request boundary in the pipeline.

Key design decisions
--------------------
1.  **API-contract preservation** — every schema mirrors the existing
    ``Form(...)`` parameters in ``app.py`` exactly.  Field names, types,
    and default values are byte-for-byte identical.  Nothing here breaks the
    existing HTTP contract.

2.  **Enums are the single source of truth** — ``Language``, ``Mode``,
    ``Department``, and ``BusinessUnit`` replace the inline ``class Enum``
    definitions scattered across ``app.py`` (lines 290–312) and
    ``patient_history.py``.  ``app.py`` will import from here once Phase 2
    wires them in.

3.  **Validators cascade from ``validators.py``** — pure helpers from the
    sibling module are used for text sanitisation so both input and output
    layers behave identically.

4.  **Pydantic v2 patterns** — ``model_validator(mode="after")`` for
    cross-field constraints; ``field_validator`` for single-field coercion.
    ``ConfigDict(extra="forbid")`` on input schemas catches typos in
    programmatic callers early.

5.  **Performance** — model class compilation happens once at import time;
    individual validation calls are O(fields) with no I/O.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schemas.validators import (
    MAX_FIELD_TEXT_LEN,
    MAX_NAME_LEN,
    MAX_PATIENT_ID_LEN,
    MAX_SESSION_ID_LEN,
    MAX_TEXT_LEN,
    sanitize_medical_text,
)


# ══════════════════════════════════════════════════════════════════════════════
# Enums  (replaces inline definitions in app.py:290-312 and patient_history.py)
# ══════════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    """Spoken / transcription language.

    Maps to the ``language`` Form field on every endpoint that accepts text.
    Values are intentionally lower-case BCP-47 codes, matching what the
    existing frontend sends.
    """
    arabic  = "ar"
    english = "en"


class Mode(str, Enum):
    """Dictation mode for ``/audio/analyze``.

    * ``doctor``       — single-speaker medical dictation; refinement is
      skipped and the transcript is passed straight through.
    * ``conversation`` — structured two-party transcript (Doctor / Patient);
      refinement adds speaker labels.
    """
    doctor       = "doctor"
    conversation = "conversation"


class Department(str, Enum):
    """Clinical note template selector.

    Each value maps 1-to-1 to a Word template and a ``DEPARTMENT_FIELDS``
    entry in ``model/department_fields.py``.
    """
    er     = "er"
    opd    = "opd"
    dental = "dental"


class BusinessUnit(str, Enum):
    """Hospital business unit that owns the patient's HCMS database.

    Each unit has its own SQL Server instance defined in ``Config.BUSINESS_UNITS``.
    """
    AHJ = "AHJ"
    AMH = "AMH"
    ANC = "ANC"


# ══════════════════════════════════════════════════════════════════════════════
# Shared base config
# ══════════════════════════════════════════════════════════════════════════════

class _StrictInputBase(BaseModel):
    """Base for all inbound request schemas.

    * ``extra="forbid"``  — rejects unknown keys on programmatic callers,
      helping catch typos early.  FastAPI Form endpoints never hit this
      because Form parameters are resolved before Pydantic sees them.
    * ``str_strip_whitespace=True`` — free, automatic edge-whitespace trim.
    * ``validate_assignment=True`` — re-validates on attribute set (useful
      in tests that mutate model instances).
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        populate_by_name=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1 ── POST /audio/analyze
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeInput(_StrictInputBase):
    """Validated input for ``POST /audio/analyze``.

    Mirrors ``analyze_transcript()`` in ``app.py`` — every Form parameter,
    type, default, and constraint is reproduced faithfully.

    The schema is intentionally not wired into ``app.py`` yet (that happens in
    Phase 2); it validates the same data shape and can be instantiated from
    ``request.form`` directly::

        data = AnalyzeInput(**dict(await request.form()))

    Cross-field invariant
    ---------------------
    ``use_history=True`` requires both ``patient_id`` **and** ``business_unit``
    to be present — the backend cannot know which database to query otherwise.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LEN,
        description="Raw transcript from live transcription (may include user edits)",
    )
    language: Language = Field(
        Language.english,
        description="Audio language",
    )
    mode: Mode = Field(
        Mode.doctor,
        description="Doctor or Conversation mode",
    )
    department: Department = Field(
        Department.opd,
        description="Clinical note template to fill: ER, OPD, or Dental",
    )
    specialty: Optional[str] = Field(
        None,
        description="Medical specialty — drives fuzzy term correction and prompt context",
    )
    use_history: bool = Field(
        False,
        description="If true, look up patient visit history and factor it into diagnosis",
    )
    patient_id: Optional[str] = Field(
        None,
        max_length=MAX_PATIENT_ID_LEN,
        pattern=r"^[A-Za-z0-9\-]+$",
        description=(
            "Patient Code (the 'Code' field on the Patient Info card) — "
            "required when use_history is true"
        ),
    )
    business_unit: Optional[BusinessUnit] = Field(
        None,
        description=(
            "AHJ, AMH or ANC — which business unit's database holds this "
            "patient's history. Required when use_history is true"
        ),
    )
    clinician_name: Optional[str] = Field(
        None,
        max_length=MAX_NAME_LEN,
        description="Self-reported doctor name — best-effort session attribution",
    )
    transcription_sec: Optional[float] = Field(
        None,
        ge=0.0,
        le=3_600.0,
        description=(
            "Client-measured duration of the transcription session (seconds). "
            "This endpoint never does transcription itself; it only logs what "
            "the browser measured."
        ),
    )
    session_id: Optional[str] = Field(
        None,
        max_length=MAX_SESSION_ID_LEN,
        description=(
            "Client-generated ID for this record→analyze→feedback cycle so a "
            "later feedback submission can be joined back to this session."
        ),
    )

    # ── Field validators ─────────────────────────────────────────────────────

    @field_validator("text", mode="before")
    @classmethod
    def _sanitize_text(cls, v: str) -> str:
        """Full Unicode-safe whitespace normalization for the main transcript."""
        return sanitize_medical_text(v, field_name="text")

    @field_validator("specialty", mode="before")
    @classmethod
    def _clean_specialty(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace; return None for blank specialty strings."""
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    @field_validator("clinician_name", mode="before")
    @classmethod
    def _clean_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    # ── Cross-field invariants ────────────────────────────────────────────────

    @model_validator(mode="after")
    def _history_requires_patient_and_bu(self) -> "AnalyzeInput":
        """``use_history=True`` is meaningless without a database target."""
        if self.use_history and (not self.patient_id or not self.business_unit):
            raise ValueError(
                "use_history requires both patient_id (Patient Code) and business_unit"
            )
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 2 ── POST /audio/refine-field
# ══════════════════════════════════════════════════════════════════════════════

class RefineFieldInput(_StrictInputBase):
    """Validated input for ``POST /audio/refine-field``.

    This endpoint refines a *single* spoken answer for one clinical-note field.
    It is intentionally leaner than ``AnalyzeInput``: no department, no
    history, no session tracking — just text + language + optional specialty.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_FIELD_TEXT_LEN,
        description="Raw transcript of the doctor's spoken answer for one field",
    )
    language: Language = Field(
        Language.english,
        description="Language the answer was spoken in",
    )
    specialty: Optional[str] = Field(
        None,
        description="Medical specialty (optional, for term correction context)",
    )

    @field_validator("text", mode="before")
    @classmethod
    def _sanitize_text(cls, v: str) -> str:
        return sanitize_medical_text(v, field_name="text")

    @field_validator("specialty", mode="before")
    @classmethod
    def _clean_specialty(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None


# ══════════════════════════════════════════════════════════════════════════════
# 3 ── WebSocket /ws/transcribe — init message
# ══════════════════════════════════════════════════════════════════════════════

class WebSocketInitInput(_StrictInputBase):
    """Validated shape of the first text message sent over ``/ws/transcribe``.

    The WebSocket protocol expects the client to send a JSON blob immediately
    after connecting::

        {"sampleRate": 16000, "language": "ar", "hotwords": "optional string"}

    ``app.py`` currently parses this with bare ``json.loads`` and ``dict.get``.
    This schema lets callers validate the blob explicitly::

        init_data = WebSocketInitInput.model_validate(json.loads(raw_text))

    Defaults mirror the existing fallback values in ``app.py`` so old clients
    that omit fields continue to work.
    """

    sampleRate: int = Field(  # noqa: N815  — camelCase to match the JS payload
        16_000,
        ge=8_000,
        le=48_000,
        description="Input PCM sample rate in Hz (browser typically sends 16000 or 48000)",
    )
    language: str = Field(
        "ar",
        pattern=r"^(ar|en)$",
        description="Transcription language: 'ar' (Munsit) or 'en' (Voxtral)",
    )
    hotwords: Optional[str] = Field(
        None,
        description="Optional comma-separated hotwords injected into the STT engine",
    )

    @field_validator("hotwords", mode="before")
    @classmethod
    def _clean_hotwords(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None


# ══════════════════════════════════════════════════════════════════════════════
# 4 ── POST /feedback
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackInput(_StrictInputBase):
    """Validated input for ``POST /feedback``.

    At least one of ``text`` (non-blank) or ``rating`` must be present —
    a submission with neither carries no information.

    ``rating`` is constrained to the three values the frontend sends:
    ``"up"``, ``"down"``, or absent (``None``).  Empty string is normalised
    to ``None`` for backward compat with old form submissions.
    """

    rating: Optional[str] = Field(
        None,
        description='"up", "down", or omitted for no rating',
    )
    text: str = Field(
        "",
        description="Free-text feedback, complaint, or enhancement suggestion",
    )
    name: Optional[str] = Field(
        None,
        max_length=MAX_NAME_LEN,
        description="Self-reported name of the person giving feedback",
    )
    department: Optional[str] = Field(
        None,
        description="Department the session was for",
    )
    language: Optional[str] = Field(
        None,
        description="Language used during the session",
    )
    session_id: Optional[str] = Field(
        None,
        max_length=MAX_SESSION_ID_LEN,
        description=(
            "The record→analyze session this feedback is about — joins back to "
            "the corresponding usage-log entry on the dashboard"
        ),
    )

    # ── Field validators ─────────────────────────────────────────────────────

    @field_validator("rating", mode="before")
    @classmethod
    def _validate_rating(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        cleaned = str(v).strip().lower()
        if cleaned not in ("up", "down"):
            raise ValueError('rating must be "up" or "down"')
        return cleaned

    @field_validator("name", "department", "language", mode="before")
    @classmethod
    def _clean_optional_str(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    # ── Cross-field invariant ─────────────────────────────────────────────────

    @model_validator(mode="after")
    def _must_have_content(self) -> "FeedbackInput":
        """Feedback with neither a text body nor a rating is useless."""
        has_text   = bool((self.text or "").strip())
        has_rating = bool(self.rating)
        if not has_text and not has_rating:
            raise ValueError("Provide feedback text and/or a thumbs rating")
        return self
