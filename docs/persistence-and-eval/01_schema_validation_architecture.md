# Schema Validation Architecture (Pydantic v2)

## 1. Overview
MedVoice processes critical clinical dictations across multiple departments (**Emergency / ER**, **Outpatient / OPD**, and **Dental**). Robust, zero-leak schema validation ensures:
- Strict validation at the API boundary before running costly LLM extraction prompts.
- Structured, machine-readable validation errors instead of unhandled exceptions.
- Consistent clinical output contracts across different language modes (Arabic, English) and specialties.

---

## 2. Input Schemas (`src/schemas/input_schemas.py`)

All API payloads conform to strict Pydantic v2 models:

```python
class BusinessUnit(str, Enum):
    AHJ = "AHJ"
    AMH = "AMH"
    ANC = "ANC"

class Department(str, Enum):
    ER = "er"
    OPD = "opd"
    DENTAL = "dental"

class PipelineMode(str, Enum):
    STREAMING = "streaming"
    BATCH = "batch"

class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., min_length=5, description="Clinical dialogue or dictation transcript")
    language: str = Field("ar", pattern=r"^(ar|en)$", description="Source audio language")
    department: Department = Field(Department.OPD, description="Target clinical department")
    specialty: Optional[str] = Field(None, description="Clinical specialty (e.g., Cardiology, Orthopedics)")
    patient_id: Optional[str] = Field(None, max_length=64, description="Hospital patient identifier")
    business_unit: Optional[BusinessUnit] = Field(None, description="Hospital business unit code")
    clinician_name: Optional[str] = Field(None, max_length=128)
```

---

## 3. Output Schemas (`src/schemas/output_schemas.py`)

Extracted clinical information is validated using department-specific models:

### 3.1 Base Clinical Extraction
```python
class ClinicalExtraction(BaseModel):
    chief_complaint: str = Field(..., description="Primary reason for visit")
    history_of_present_illness: Optional[str] = None
    past_medical_history: List[str] = Field(default_factory=list)
    medications: List[Dict[str, Any]] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vital_signs: Optional[Dict[str, Any]] = None
    physical_examination: Optional[Dict[str, Any]] = None
    assessment_and_plan: Optional[str] = None
```

### 3.2 Department Specializations
- **Dental (`DentalExtraction`)**: Adds fields for tooth numbering systems (Palmer/FDI), periodontal assessment, plaque index, and restorative plans.
- **Emergency (`ERExtraction`)**: Adds triage acuity category (ESI 1-5), red flag indicators, rapid disposition, and trauma survey.

---

## 4. Schema Validator Utilities (`src/schemas/validators.py`)

To maximize performance, Pydantic model validators are compiled once and reused across requests. Pre-validation cleanses input transcripts, removing invalid Unicode control characters or accidental prompt-injection payloads while preserving Arabic and English medical vocabulary.
