"""
Typed data models and type definitions for MongoDB collections.
Enforces clean schema contracts across:
- sessions
- pipeline_runs
- feedback
- gold_cases
- eval_runs
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ==============================================================================
# TypedDict representations for MongoDB documents
# ==============================================================================

class SessionDoc(TypedDict, total=False):
    session_id: str
    created_at: datetime
    updated_at: Optional[datetime]
    ip: Optional[str]
    user_agent: Optional[str]
    clinician_name: Optional[str]
    sso_user: Optional[str]
    language: str
    mode: str
    department: str
    specialty: Optional[str]
    patient_id: Optional[str]
    business_unit: Optional[str]
    status: str
    total_sec: Optional[float]
    transcription_sec: Optional[float]
    refinement_sec: Optional[float]
    translation_sec: Optional[float]
    extraction_sec: Optional[float]
    question_generation_sec: Optional[float]
    diagnosis_sec: Optional[float]
    transcript: Optional[str]
    refined_text: Optional[str]
    translated_text: Optional[str]
    patient_story: Optional[str]
    extracted_features: Optional[Dict[str, Any]]
    questions: Optional[List[Dict[str, Any]]]
    diagnosis: Optional[Dict[str, Any]]
    extraction_usage: Optional[Dict[str, Any]]
    diagnosis_usage: Optional[Dict[str, Any]]
    history_check: Optional[Dict[str, Any]]


class PipelineRunDoc(TypedDict, total=False):
    session_id: str
    step_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timings: Dict[str, Any]
    usage_meta: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: datetime


class FeedbackDoc(TypedDict, total=False):
    session_id: Optional[str]
    rating: Optional[str]
    text: Optional[str]
    name: Optional[str]
    ip: Optional[str]
    sso_user: Optional[str]
    department: Optional[str]
    language: Optional[str]
    created_at: datetime


class GoldCaseDoc(TypedDict, total=False):
    case_id: str
    department: str
    specialty: Optional[str]
    audio_language: str
    transcript: str
    expected_extraction: Dict[str, Any]
    expected_diagnosis: Optional[Dict[str, Any]]
    expected_questions: Optional[List[str]]
    source: Optional[str]
    created_at: datetime
    updated_at: datetime


class EvalRunDoc(TypedDict, total=False):
    eval_id: str
    case_id: str
    session_id: Optional[str]
    created_at: datetime
    model_id: str
    actual_extraction: Optional[Dict[str, Any]]
    actual_diagnosis: Optional[Dict[str, Any]]
    scores: Optional[Dict[str, Any]]
    field_diffs: Optional[Dict[str, Any]]


# ==============================================================================
# Pydantic Schema Models for Gold Cases & Eval Runs (Phase 3 & Phase 4 readiness)
# ==============================================================================

class GoldCaseModel(BaseModel):
    case_id: str = Field(..., description="Unique gold test case ID")
    department: str = Field(..., description="er | opd | dental")
    specialty: Optional[str] = None
    audio_language: str = Field("ar", description="Language of clinical case")
    transcript: str = Field(..., description="Reference clinical transcript")
    expected_extraction: Dict[str, Any] = Field(default_factory=dict)
    expected_diagnosis: Optional[Dict[str, Any]] = None
    expected_questions: Optional[List[str]] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class EvalRunModel(BaseModel):
    eval_id: str = Field(..., description="Unique evaluation execution ID")
    case_id: str = Field(..., description="Linked gold case ID")
    session_id: Optional[str] = None
    model_id: str = Field(..., description="Target model ID evaluated")
    actual_extraction: Optional[Dict[str, Any]] = None
    actual_diagnosis: Optional[Dict[str, Any]] = None
    scores: Dict[str, Any] = Field(default_factory=dict)
    field_diffs: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
