"""
Ground-truth clinical case schemas for evaluation and benchmark persistence.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class GoldCase(BaseModel):
    """One ground-truth clinical case for evaluation."""
    case_id: str = Field(..., description="Unique ID, e.g. 'ER-1', 'OPD-101'")
    department: str = Field(..., pattern=r"^(er|opd|dental)$", description="er | opd | dental")
    specialty: Optional[str] = None
    audio_language: str = Field("ar", pattern=r"^(ar|en)$", description="ar | en")

    # Input transcript
    transcript: str = Field(..., min_length=10, description="Raw clinical dictation/dialogue")

    # Expected outputs
    expected_extraction: Dict[str, Any] = Field(default_factory=dict, description="Expected clinical fields")
    expected_diagnosis: Optional[Dict[str, Any]] = Field(None, description="Expected diagnosis object / ICD10 codes")
    expected_questions: Optional[List[Dict[str, Any]]] = Field(None, description="Expected clarifying questions")

    # Provenance
    source: str = Field("manual", description="manual | xlsx_import | clinician_review")
    annotator: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    is_active: bool = True
    tags: List[str] = Field(default_factory=list)

    @field_validator("department", mode="before")
    @classmethod
    def _norm_dept(cls, v: Any) -> str:
        return str(v).strip().lower() if v else ""

    @field_validator("audio_language", mode="before")
    @classmethod
    def _norm_lang(cls, v: Any) -> str:
        return str(v).strip().lower() if v else "ar"
