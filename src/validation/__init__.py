"""
MedVoice Validation Subsystem - Public API
==========================================
Exporting validation contracts, schemas, consistency rules, history guards,
missing-field checklists, and the ClinicalValidationPipeline orchestrator.
"""

from validation.consistency import (
    CautionBanner,
    CautionSeverity,
    ConsistencyRuleType,
    ConsistencyValidator,
)
from validation.exceptions import (
    ClinicalSchemaValidationError,
    FieldValidationError,
    SchemaRejectionPayload,
    ValidationErrorCode,
)
from validation.history_guard import (
    HistoryAlignmentReport,
    HistoryCrossCheckGuard,
    HistoryDiscrepancyFlag,
)
from validation.icd10 import (
    BatchValidationResult,
    CodeValidationResult,
    ICD10Validator,
    RejectionReason,
    ResolutionTier,
    get_icd10_validator,
)
from validation.laterality import (
    Laterality,
    LateralityDetection,
    check_laterality_conflict,
    detect_laterality,
)
from validation.missing_fields import (
    MissingFieldChecklist,
    MissingFieldItem,
    MissingFieldsReport,
)
from validation.pipeline import (
    ClinicalValidationPipeline,
    ValidationReport,
)
from validation.schemas import (
    ClinicalExtractionBase,
    DentalExtractionSchema,
    DiagnosisReasoningSchema,
    ERExtractionSchema,
    OPDExtractionSchema,
    SchemaValidator,
    get_extraction_schema_for_department,
)

__all__ = [
    # Pipeline Orchestrator
    "ClinicalValidationPipeline",
    "ValidationReport",
    # ICD-10
    "ICD10Validator",
    "get_icd10_validator",
    "CodeValidationResult",
    "BatchValidationResult",
    "RejectionReason",
    "ResolutionTier",
    # Schemas
    "ClinicalExtractionBase",
    "ERExtractionSchema",
    "OPDExtractionSchema",
    "DentalExtractionSchema",
    "DiagnosisReasoningSchema",
    "SchemaValidator",
    "get_extraction_schema_for_department",
    # Consistency & Laterality
    "ConsistencyValidator",
    "CautionBanner",
    "CautionSeverity",
    "ConsistencyRuleType",
    "Laterality",
    "LateralityDetection",
    "detect_laterality",
    "check_laterality_conflict",
    # History Cross-Check
    "HistoryCrossCheckGuard",
    "HistoryAlignmentReport",
    "HistoryDiscrepancyFlag",
    # Missing Fields
    "MissingFieldChecklist",
    "MissingFieldItem",
    "MissingFieldsReport",
    # Exceptions & Errors
    "ValidationErrorCode",
    "FieldValidationError",
    "SchemaRejectionPayload",
    "ClinicalSchemaValidationError",
]
