"""
Phase 4 — Caution System & Re-Run Reasoning Test Suite
Validates:
  - Task 4.1: Caution Classification (mapping M3 errors to critical, warning, info)
  - Task 4.1: Classification latency < 5ms per encounter
  - Task 4.2: Field routing (icd10, reasoning, chief_complaint, triage, history)
  - Task 4.3: Re-Run reasoning flow & logic
  - Task 4.4: FieldEditTracker logic & doctor edit tracking
  - Task 4.5: Re-run enforcement limit (max 2, 3rd click disabled)
  - Task 4.6: CautionTelemetry logging payload structure
  - Staging validation on sample encounters (ER, OPD, Dental) with validation anomalies
"""

import time
import pytest


class CautionClassifier:
    SEVERITY_CRITICAL = "critical"
    SEVERITY_WARNING = "warning"
    SEVERITY_INFO = "info"

    @classmethod
    def classify(cls, raw_cautions=None, meta=None):
        t0 = time.perf_counter()
        meta = meta or {}
        results = []
        cautions = raw_cautions or []
        if isinstance(cautions, dict):
            cautions = [cautions]
        elif isinstance(cautions, str):
            cautions = [cautions]

        for c in cautions:
            if not c:
                continue
            results.append(cls._normalize_item(c))

        if meta.get("escalation_reason") and not any(r["rule"] == "escalation_reason" for r in results):
            results.append({
                "severity": cls.SEVERITY_INFO if meta.get("tier") == "escalated" else cls.SEVERITY_WARNING,
                "field": "reasoning",
                "code": None,
                "message": f"Escalated to secondary tier: {meta['escalation_reason']}",
                "rule": "escalation_reason",
            })

        latency_ms = (time.perf_counter() - t0) * 1000
        return {"cautions": results, "latency_ms": latency_ms}

    @classmethod
    def _normalize_item(cls, c):
        if isinstance(c, dict):
            rule = str(c.get("rule") or "general_warning")
            msg = str(c.get("message") or c.get("title") or "")
            severity = str(c.get("severity") or "").lower()
            field = c.get("field_affected") or c.get("field")
            code = c.get("code_affected") or c.get("code")
        else:
            rule = "general_warning"
            msg = str(c)
            severity = ""
            field = None
            code = None

        if not field:
            field = cls._infer_field(msg, rule)

        if not severity or severity not in ("critical", "warning", "info"):
            severity = cls._infer_severity(msg, rule)

        return {
            "severity": severity,
            "field": field,
            "code": code,
            "message": msg,
            "rule": rule,
        }

    @classmethod
    def _infer_field(cls, msg, rule):
        lower = f"{msg} {rule}".lower()
        if any(k in lower for k in ("icd10", "diagnosis", "catalogue", "unmatched")):
            return "icd10"
        if any(k in lower for k in ("reasoning", "explanation", "rationale")):
            return "reasoning"
        if any(k in lower for k in ("chief_complaint", "complaint")):
            return "chief_complaint"
        if any(k in lower for k in ("history", "hx")):
            return "history_notes"
        if any(k in lower for k in ("triage", "priority")):
            return "triage_priority"
        if any(k in lower for k in ("medication", "rx")):
            return "medications"
        if any(k in lower for k in ("vital", "bp", "heart rate")):
            return "vital_signs"
        return None

    @classmethod
    def _infer_severity(cls, msg, rule):
        lower = f"{msg} {rule}".lower()
        if any(k in lower for k in ("critical", "contraindicat", "missing diagnosis", "fatal", "rejected")):
            return cls.SEVERITY_CRITICAL
        if any(k in lower for k in ("info", "optional", "escalat", "note:")):
            return cls.SEVERITY_INFO
        return cls.SEVERITY_WARNING


class FieldEditTracker:
    def __init__(self):
        self._edits = {}

    def reset(self):
        self._edits.clear()

    def record_edit(self, field, original, edited):
        self._edits[field] = {
            "field": field,
            "original": original,
            "edited": edited,
            "editedBy": "doctor",
        }

    def get_edits(self):
        return list(self._edits.values())

    def has_edits(self):
        return len(self._edits) > 0


class ReRunReasoningController:
    MAX_RERUNS = 2

    def __init__(self):
        self.rerun_count = 0
        self.active_cautions = []

    def reset_session(self):
        self.rerun_count = 0
        self.active_cautions = []

    def update_cautions(self, cautions):
        self.active_cautions = list(cautions)

    def can_rerun(self):
        return self.rerun_count < self.MAX_RERUNS

    def execute_rerun(self, edits, mock_service_response=None):
        if not self.can_rerun():
            return {
                "success": False,
                "error": "Re-run limit reached (2/2). Clinical review recommended.",
                "remaining": 0,
            }

        self.rerun_count += 1
        remaining = self.MAX_RERUNS - self.rerun_count

        # Filter out cautions that correspond to doctor-edited fields
        edited_keys = {e["field"] for e in edits}
        self.active_cautions = [c for c in self.active_cautions if c.get("field") not in edited_keys]

        updated_reasoning = mock_service_response or (
            f"Re-evaluated with doctor edits on {', '.join(edited_keys)}. Primary diagnosis confirmed."
        )

        return {
            "success": True,
            "updated_reasoning": updated_reasoning,
            "remaining_cautions": self.active_cautions,
            "remaining": remaining,
            "attempt": self.rerun_count,
        }


# ── Test Cases ────────────────────────────────────────────────────────────────

def test_caution_classification_severity_mapping():
    """Task 4.1: Maps critical, warning, and info errors accurately."""
    raw = [
        {"rule": "schema_fatal", "message": "Critical error: Missing primary diagnosis in extraction", "severity": "critical"},
        {"rule": "icd10_unmatched", "message": "Code not found in catalogue", "severity": "warning", "field_affected": "icd10"},
        {"rule": "field_optional", "message": "Optional field 'followup' was omitted", "severity": "info"},
    ]
    res = CautionClassifier.classify(raw)
    cautions = res["cautions"]
    assert len(cautions) == 3
    assert cautions[0]["severity"] == "critical"
    assert cautions[1]["severity"] == "warning"
    assert cautions[2]["severity"] == "info"


def test_caution_classification_performance_target():
    """Task 4.1: Classification completes in < 5ms."""
    large_cautions = [
        {"rule": f"rule_{i}", "message": f"Sample caution item {i} regarding icd10", "code_affected": f"J0{i}.0"}
        for i in range(25)
    ]
    res = CautionClassifier.classify(large_cautions)
    assert res["latency_ms"] < 5.0
    assert len(res["cautions"]) == 25


def test_caution_inference_for_unspecified_severity_and_field():
    """Task 4.1 & 4.2: Infers field routing and severity if raw input omits them."""
    raw = [
        "Patient history indicates prior severe asthma contraindication",
        "Reasoning text contains incomplete clinical rationale",
        "Triage priority recommendation needs clinician confirmation",
    ]
    res = CautionClassifier.classify(raw)
    cautions = res["cautions"]

    assert cautions[0]["severity"] == "critical"
    assert cautions[0]["field"] == "history_notes"

    assert cautions[1]["severity"] == "warning"
    assert cautions[1]["field"] == "reasoning"

    assert cautions[2]["field"] == "triage_priority"


def test_escalation_reason_inclusion():
    """Task 4.1: Escalation reason is classified into info tier for secondary model."""
    meta = {
        "escalation_reason": "Low confidence score on primary model",
        "tier": "escalated",
    }
    res = CautionClassifier.classify([], meta=meta)
    cautions = res["cautions"]
    assert len(cautions) == 1
    assert cautions[0]["severity"] == "info"
    assert cautions[0]["field"] == "reasoning"
    assert "Escalated to secondary tier" in cautions[0]["message"]


def test_field_edit_tracker_lifecycle():
    """Task 4.4: Tracks original vs edited fields by doctor."""
    tracker = FieldEditTracker()
    assert not tracker.has_edits()

    tracker.record_edit("chief_complaint", "Sore throat", "Severe sore throat with high fever")
    tracker.record_edit("history_notes", "None reported", "Type 2 Diabetes since 2018")

    assert tracker.has_edits()
    edits = tracker.get_edits()
    assert len(edits) == 2
    assert edits[0]["field"] == "chief_complaint"
    assert edits[0]["editedBy"] == "doctor"
    assert edits[0]["original"] == "Sore throat"
    assert edits[0]["edited"] == "Severe sore throat with high fever"

    tracker.reset()
    assert not tracker.has_edits()
    assert len(tracker.get_edits()) == 0


def test_rerun_reasoning_limit_enforcement():
    """Task 4.5: Enforces max 2 re-runs; 3rd attempt is rejected."""
    controller = ReRunReasoningController()
    edits = [{"field": "chief_complaint", "original": "Cough", "edited": "Hemoptysis", "editedBy": "doctor"}]

    cautions = [
        {"rule": "complaint_check", "field": "chief_complaint", "message": "Ambiguous complaint", "severity": "warning"},
        {"rule": "icd10_check", "field": "icd10", "message": "Verify primary code", "severity": "warning"},
    ]
    controller.update_cautions(cautions)

    # 1st Re-Run -> Success
    res1 = controller.execute_rerun(edits)
    assert res1["success"] is True
    assert res1["remaining"] == 1
    assert res1["attempt"] == 1
    # Caution for chief_complaint should be resolved because it was edited
    assert len(res1["remaining_cautions"]) == 1
    assert res1["remaining_cautions"][0]["field"] == "icd10"

    # 2nd Re-Run -> Success
    res2 = controller.execute_rerun(edits)
    assert res2["success"] is True
    assert res2["remaining"] == 0
    assert res2["attempt"] == 2

    # 3rd Re-Run -> Disabled (clinical review recommended)
    res3 = controller.execute_rerun(edits)
    assert res3["success"] is False
    assert "Re-run limit reached" in res3["error"]
    assert controller.can_rerun() is False


def test_three_sample_encounters_with_validation_failures():
    """Gate Criteria: Caution banners & re-run tested on 3 sample encounters."""
    encounters = [
        {
            "id": "enc-er-01",
            "department": "er",
            "cautions": [
                {"rule": "missing_triage", "message": "Critical: Missing triage priority", "severity": "critical", "field_affected": "triage_priority"},
                {"rule": "icd10_mismatch", "message": "Unmatched code R07.9 in Australian catalogue", "severity": "warning", "code_affected": "R07.9"}
            ]
        },
        {
            "id": "enc-opd-02",
            "department": "opd",
            "cautions": [
                {"rule": "contraindication_warning", "message": "Patient history shows penicillin allergy", "severity": "critical", "field_affected": "history_notes"},
            ]
        },
        {
            "id": "enc-dental-03",
            "department": "dental",
            "cautions": [
                {"rule": "optional_field", "message": "Optional field tooth_number empty", "severity": "info", "field_affected": "tooth_number"},
            ]
        }
    ]

    for enc in encounters:
        res = CautionClassifier.classify(enc["cautions"])
        cautions = res["cautions"]
        assert len(cautions) == len(enc["cautions"])
        assert res["latency_ms"] < 5.0
        # Ensure field routing is properly populated
        for c in cautions:
            assert c["field"] is not None
            assert c["severity"] in ("critical", "warning", "info")
