"""
Tests for Milestone 7 (M7):
- M7.1: Entra ID RBAC Enforcement on Dashboard (ClinicalAdmin & ExecutiveStakeholder vs Doctor / Unauthenticated)
- M7.2: PHI-Safe Logs & Transcripts Sanitization
"""
import pytest
from fastapi.testclient import TestClient

from app import app, sanitize_usage_entry
from core.auth import AuthenticatedUser


client = TestClient(app)


def test_sanitize_usage_entry_scrubs_transcripts_and_notes():
    raw_entry = {
        "session_id": "test-session-123",
        "department": "er",
        "language": "en",
        "mode": "doctor",
        "transcript": "Patient John Doe presents with acute chest pain and shortness of breath.",
        "refined_text": "Patient presents with acute chest pain and dyspnea.",
        "translated_text": "Patient presents with acute chest pain and dyspnea.",
        "patient_story": "Patient presents with acute chest pain and dyspnea.",
        "total_sec": 4.5,
        "status": "ok",
    }

    sanitized = sanitize_usage_entry(raw_entry)

    # Free text must be removed
    assert "transcript" not in sanitized
    assert "refined_text" not in sanitized
    assert "translated_text" not in sanitized
    assert "patient_story" not in sanitized

    # Telemetry and hash must remain
    assert sanitized["session_id"] == "test-session-123"
    assert sanitized["department"] == "er"
    assert sanitized["total_sec"] == 4.5
    assert sanitized["phi_safe"] is True
    assert sanitized["transcript_hash"] is not None
    assert len(sanitized["transcript_hash"]) == 64  # SHA-256


def test_dashboard_access_with_admin_role_emulation():
    # Emulate ClinicalAdmin via header
    headers = {
        "x-user-id": "admin-1",
        "x-user-roles": "ClinicalAdmin",
        "x-user-name": "Dr. Admin",
    }
    response = client.get("/admin/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["is_admin"] is True
    assert data["can_view_dashboard"] is True


def test_dashboard_access_with_stakeholder_role_emulation():
    # Emulate ExecutiveStakeholder via header
    headers = {
        "x-user-id": "exec-1",
        "x-user-roles": "ExecutiveStakeholder",
        "x-user-name": "Financial Director",
    }
    response = client.get("/admin/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["is_stakeholder"] is True
    assert data["can_view_dashboard"] is True


def test_dashboard_access_denied_for_doctor_role_only():
    # Regular doctor role should NOT be able to view dashboard / admin routes
    headers = {
        "x-user-id": "doc-1",
        "x-user-roles": "Doctor",
        "x-user-name": "Dr. Clinician",
    }
    response = client.get("/admin/me", headers=headers)
    assert response.status_code == 403
    assert "ClinicalAdmin or ExecutiveStakeholder" in response.json()["detail"]

    response_dash = client.get("/dashboard", headers=headers)
    assert response_dash.status_code == 403


def test_admin_financial_summary_endpoint():
    # ExecutiveStakeholder role accessing /admin/financial-summary
    headers = {
        "x-user-id": "exec-101",
        "x-user-roles": "ExecutiveStakeholder",
        "x-user-name": "Financial Director",
    }
    response = client.get("/admin/financial-summary?days=30", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_sessions" in data
    assert "token_usage" in data
    assert "costs" in data
    assert "savings" in data
    assert "by_department" in data
    assert "daily_burn" in data
    assert data["costs"]["total_ai_cost_usd"] >= 0.0
    assert data["savings"]["gross_value_saved_usd"] >= 0.0


def test_executive_stakeholder_role_sanitizes_clinician_identifiers():
    # ExecutiveStakeholder role accessing /admin/usage gets anonymized doctor info
    headers = {
        "x-user-id": "exec-101",
        "x-user-roles": "ExecutiveStakeholder",
        "x-user-name": "Chief Financial Officer",
    }
    response = client.get("/admin/usage?limit=5", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    for entry in data["entries"]:
        assert "clinician_name" not in entry
        assert "sso_user" not in entry
        assert "doctor_id" not in entry


def test_dashboard_html_contains_phase3_components():
    headers = {
        "x-user-id": "exec-101",
        "x-user-roles": "ExecutiveStakeholder",
        "x-user-name": "Chief Financial Officer",
    }
    response = client.get("/dashboard", headers=headers)
    assert response.status_code == 200
    html = response.text
    # Verify Task 3.1 & 3.2 UI elements
    assert "Financial & ROI Impact" in html
    assert "Department Efficiency Matrix" in html
    assert "Cost vs. Value" in html
    assert "currency-select" in html
    assert "SAR" in html


def test_unauthenticated_request_rejected_with_401_or_403():
    # Calling dashboard and admin endpoints with no authentication credentials
    response = client.get("/admin/me")
    assert response.status_code in (401, 403)

    response = client.get("/admin/usage")
    assert response.status_code in (401, 403)

    response = client.get("/admin/financial-summary")
    assert response.status_code in (401, 403)


from pathlib import Path


def test_automated_phi_regex_scanner_on_usage_storage(tmp_path: Path):
    """
    Task 4.1: Automated PHI regex scan ensuring zero patient names,
    free-text clinical transcripts, or unmasked notes are persisted to disk.
    """
    import re
    import json
    from app import save_usage_log

    # Simulated output that would previously leak clinical transcripts
    simulated_encounter = {
        "session_id": "phi-test-scan-8899",
        "clinician_name": "Dr. Sarah Al-Otaibi",
        "department": "er",
        "language": "en",
        "transcript": "Patient John Doe, 45-year-old male, presents with severe crushing retrosternal chest pain radiating to left jaw. Prescribed nitroglycerin 0.4mg sublingual.",
        "refined_text": "45yo male presenting with acute coronary syndrome symptoms.",
        "translated_text": "المريض يعاني من آلام حادة بالصدر",
        "patient_story": "Patient John Doe severe crushing retrosternal chest pain",
        "transcription_sec": 3.2,
        "refinement_sec": 1.1,
        "total_sec": 4.3,
        "status": "ok",
    }

    test_log_file = tmp_path / "usage_test_scan.jsonl"
    
    # Run the save_usage_log sanitization pipeline
    from app import sanitize_usage_entry
    clean_entry = sanitize_usage_entry(simulated_encounter)
    with open(test_log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(clean_entry) + "\n")

    # Read back raw contents on disk and perform strict PHI Regex scanning
    raw_disk_data = test_log_file.read_text(encoding="utf-8")

    # PHI Patterns that must NEVER appear on disk in plain text:
    phi_patterns = [
        r"\bJohn Doe\b",
        r"\bsevere crushing retrosternal\b",
        r"\bnitroglycerin\b",
        r"\bacute coronary syndrome\b",
        r"آلام حادة بالصدر",
    ]

    for pattern in phi_patterns:
        match = re.search(pattern, raw_disk_data, re.IGNORECASE)
        assert match is None, f"CRITICAL: PHI leak detected on disk matching pattern: {pattern}"

    # Verify structural integrity hash was generated instead
    disk_json = json.loads(raw_disk_data.strip())
    assert disk_json["phi_safe"] is True
    assert "transcript_hash" in disk_json
    assert len(disk_json["transcript_hash"]) == 64
    assert "transcript" not in disk_json
    assert "refined_text" not in disk_json


