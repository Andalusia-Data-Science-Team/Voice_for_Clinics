"""
MedVoice Validation Subsystem - Validation Exceptions & Error Taxonomy
========================================================================
Part of Phase 2 (Milestone 3.2): Strict Pydantic Output Schemas & Validation Contracts.

Provides structured, machine-actionable validation error definitions:
  - Structured location paths ("loc")
  - Standard error codes
  - Human and machine-readable rejection reasons
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ValidationErrorCode(str, Enum):
    """Standardized validation error codes."""
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FIELD_TYPE = "INVALID_FIELD_TYPE"
    INVALID_DEPARTMENT = "INVALID_DEPARTMENT"
    INVALID_DIAGNOSIS_OUTPUT = "INVALID_DIAGNOSIS_OUTPUT"
    UNEXPECTED_DATA_SHAPE = "UNEXPECTED_DATA_SHAPE"


@dataclass(frozen=True, slots=True)
class FieldValidationError:
    """Represents a specific field validation failure."""
    loc: tuple[str, ...]
    reason: str
    error_code: str = ValidationErrorCode.SCHEMA_VALIDATION_FAILED.value
    input_value: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loc": list(self.loc),
            "reason": self.reason,
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class SchemaRejectionPayload:
    """Standardized rejection payload returned when schema validation fails."""
    status: str = "rejected"
    error_code: str = ValidationErrorCode.SCHEMA_VALIDATION_FAILED.value
    reason: str = "Schema validation failed"
    loc: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error_code": self.error_code,
            "reason": self.reason,
            "loc": self.loc,
            "errors": self.errors,
        }


class ClinicalSchemaValidationError(ValueError):
    """Raised when clinical LLM output fails schema validation."""

    def __init__(
        self,
        reason: str,
        loc: Optional[Sequence[str]] = None,
        error_code: str = ValidationErrorCode.SCHEMA_VALIDATION_FAILED.value,
        field_errors: Optional[list[FieldValidationError]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.loc = list(loc or [])
        self.error_code = error_code
        self.field_errors = field_errors or []

    def to_rejection_payload(self) -> SchemaRejectionPayload:
        return SchemaRejectionPayload(
            status="rejected",
            error_code=self.error_code,
            reason=self.reason,
            loc=self.loc,
            errors=[err.to_dict() for err in self.field_errors],
        )
