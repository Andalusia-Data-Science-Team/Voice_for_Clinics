"""Pipeline Data Transfer Objects and Unified Merger.

Combines extraction and diagnosis outputs into a structured MergedResult
and provides data classes for step results and pipeline context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionStepResult:
    """Output from the feature extraction step."""
    features: Dict[str, Any] = field(default_factory=dict)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    extraction_sec: float = 0.0
    question_generation_sec: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DiagnosisStepResult:
    """Output from the diagnosis recommendation step."""
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    diagnosis_sec: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    history_check: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class MergedResult:
    """Unified representation of post-transcription pipeline outputs."""
    session_id: str
    department: str
    language: str
    mode: str
    specialty: Optional[str] = None
    patient_id: Optional[str] = None
    business_unit: Optional[str] = None
    use_history: bool = False
    
    # Text progression
    transcript: str = ""
    refined_text: Optional[str] = None
    translated_text: Optional[str] = None
    patient_story: str = ""
    
    # Clinical Signals
    routing_signal: Optional[Dict[str, Any]] = None
    routing_audit: Optional[Dict[str, Any]] = None
    
    # Extraction output
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    extraction_model: str = ""
    extraction_usage: Optional[Dict[str, Any]] = None
    
    # Diagnosis output
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    reasoning_model: str = ""
    diagnosis_usage: Optional[Dict[str, Any]] = None
    
    # History integration
    history_used: bool = False
    history_visit_count: int = 0
    history_error: Optional[str] = None
    history_check: Optional[Dict[str, Any]] = None
    
    # Latencies (seconds)
    transcription_sec: Optional[float] = None
    refinement_sec: Optional[float] = None
    translation_sec: Optional[float] = None
    extraction_sec: Optional[float] = None
    question_generation_sec: Optional[float] = None
    diagnosis_sec: Optional[float] = None
    total_sec: Optional[float] = None
    
    # Validation & Escalation (Phase 2 / M4.1 + M4.2 + B4.2/B4.3)
    validation_report: Optional[Dict[str, Any]] = None
    validation_passed: bool = True
    tier: str = "primary"
    escalation_reason: Optional[str] = None
    
    # Execution metadata
    status: str = "ok"
    errors: Dict[str, str] = field(default_factory=dict)

    def to_log_dict(
        self,
        ip: str = "unknown",
        user_agent: str = "unknown",
        clinician_name: Optional[str] = None,
        sso_user: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert merged result into standard usage log dictionary format."""
        from datetime import datetime, timezone
        
        return {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "ip": ip,
            "user_agent": user_agent,
            "clinician_name": clinician_name,
            "sso_user": sso_user,
            "language": self.language,
            "mode": self.mode,
            "department": self.department,
            "specialty": self.specialty,
            "patient_id": self.patient_id,
            "business_unit": self.business_unit,
            "use_history": self.use_history,
            "history_visit_count": self.history_visit_count,
            "history_error": self.history_error,
            "status": self.status,
            "total_sec": self.total_sec,
            "transcription_sec": self.transcription_sec,
            "refinement_sec": self.refinement_sec,
            "translation_sec": self.translation_sec,
            "extraction_sec": self.extraction_sec,
            "question_generation_sec": self.question_generation_sec,
            "transcript": self.transcript,
            "refined_text": self.refined_text,
            "refinement_skipped": self.refinement_sec == 0.0 and self.refined_text == self.transcript,
            "translated_text": self.translated_text,
            "extracted_features": self.extracted_features,
            "routing_signal": self.routing_signal,
            "routing_audit": self.routing_audit,
            "questions": self.questions,
            "extraction_model": self.extraction_model,
            "reasoning_model": self.reasoning_model,
            "diagnosis_sec": self.diagnosis_sec,
            "diagnosis": self.diagnosis,
            "extraction_usage": self.extraction_usage,
            "diagnosis_usage": self.diagnosis_usage,
            "history_check": self.history_check,
            "tier": self.tier,
            "escalation_reason": self.escalation_reason,
            "validation_report": self.validation_report,
        }


def merge_step_results(
    extraction_res: Optional[ExtractionStepResult],
    diagnosis_res: Optional[DiagnosisStepResult],
    session_id: str,
    department: str,
    language: str,
    mode: str,
    transcript: str,
    refined_text: Optional[str] = None,
    translated_text: Optional[str] = None,
    patient_story: str = "",
    routing_signal: Optional[Dict[str, Any]] = None,
    routing_audit: Optional[Dict[str, Any]] = None,
    specialty: Optional[str] = None,
    patient_id: Optional[str] = None,
    business_unit: Optional[str] = None,
    use_history: bool = False,
    history_visit_count: int = 0,
    history_error: Optional[str] = None,
    extraction_model: str = "",
    reasoning_model: str = "",
    refinement_sec: Optional[float] = None,
    translation_sec: Optional[float] = None,
    transcription_sec: Optional[float] = None,
    total_sec: Optional[float] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    validation_passed: bool = True,
    tier: str = "primary",
    escalation_reason: Optional[str] = None,
) -> MergedResult:
    """Create a unified MergedResult from individual step outputs."""
    errors: Dict[str, str] = {}
    
    extracted_features = {}
    questions = []
    extraction_sec = None
    question_generation_sec = None
    extraction_usage = None
    if extraction_res:
        if extraction_res.error:
            errors["extraction"] = extraction_res.error
        else:
            extracted_features = extraction_res.features
            questions = extraction_res.questions
            extraction_sec = extraction_res.extraction_sec
            question_generation_sec = extraction_res.question_generation_sec
            extraction_usage = extraction_res.meta

    diagnosis_data = {}
    diagnosis_sec = None
    diagnosis_usage = None
    history_check = None
    if diagnosis_res:
        if diagnosis_res.error:
            errors["diagnosis"] = diagnosis_res.error
        else:
            diagnosis_data = diagnosis_res.diagnosis
            diagnosis_sec = diagnosis_res.diagnosis_sec
            diagnosis_usage = diagnosis_res.meta
            history_check = diagnosis_res.history_check

    status = "error" if errors else "ok"
    
    return MergedResult(
        session_id=session_id,
        department=department,
        language=language,
        mode=mode,
        specialty=specialty,
        patient_id=patient_id,
        business_unit=business_unit,
        use_history=use_history,
        transcript=transcript,
        refined_text=refined_text,
        translated_text=translated_text,
        patient_story=patient_story,
        routing_signal=routing_signal,
        routing_audit=routing_audit,
        extracted_features=extracted_features,
        questions=questions,
        extraction_model=extraction_model,
        extraction_usage=extraction_usage,
        diagnosis=diagnosis_data,
        reasoning_model=reasoning_model,
        diagnosis_usage=diagnosis_usage,
        history_used=bool(patient_story and history_visit_count > 0 and not history_error),
        history_visit_count=history_visit_count,
        history_error=history_error,
        history_check=history_check,
        transcription_sec=transcription_sec,
        refinement_sec=refinement_sec,
        translation_sec=translation_sec,
        extraction_sec=extraction_sec,
        question_generation_sec=question_generation_sec,
        diagnosis_sec=diagnosis_sec,
        total_sec=total_sec,
        validation_report=validation_report,
        validation_passed=validation_passed,
        tier=tier,
        escalation_reason=escalation_reason,
        status=status,
        errors=errors,
    )
