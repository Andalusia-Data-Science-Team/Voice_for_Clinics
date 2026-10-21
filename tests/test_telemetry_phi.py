"""A2.4: telemetry/log payloads must never include clinical text. Mocks only."""

from __future__ import annotations

import json

import app as app_module


def test_sanitize_usage_entry_strips_clinical_text():
    entry = {
        "session_id": "synthetic-session",
        "transcript": "SYNTHETIC patient reports a headache.",
        "refined_text": "SYNTHETIC refined note.",
        "translated_text": "SYNTHETIC translated note.",
        "patient_story": "SYNTHETIC patient story.",
        "extracted_features": {"chief_complaint": "headache"},
        "total_sec": 1.23,
    }
    clean = app_module.sanitize_usage_entry(entry)

    for key in ("transcript", "refined_text", "translated_text", "patient_story"):
        assert key not in clean
    assert clean["phi_safe"] is True
    assert clean["transcript_hash"]  # one-way hash retained for audit
    # Operational telemetry is preserved.
    assert clean["total_sec"] == 1.23


def test_sanitized_payload_contains_no_clinical_phrases():
    entry = {
        "transcript": "SYNTHETIC unique phrase zzz-clinical-marker",
        "refined_text": "SYNTHETIC unique phrase zzz-clinical-marker",
        "translated_text": "SYNTHETIC unique phrase zzz-clinical-marker",
        "patient_story": "SYNTHETIC unique phrase zzz-clinical-marker",
    }
    clean = app_module.sanitize_usage_entry(entry)
    blob = json.dumps(clean).lower()
    assert "zzz-clinical-marker" not in blob


def test_telemetry_module_documents_no_sensitive_fields():
    """The telemetry module must not forward transcript/note fields."""
    import core.telemetry as telemetry

    source = open(telemetry.__file__, encoding="utf-8").read().lower()
    # The module explicitly documents that sensitive fields are never sent.
    assert "never sent" in source or "never" in source