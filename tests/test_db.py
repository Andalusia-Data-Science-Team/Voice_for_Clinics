"""
tests/test_db.py
================
Comprehensive test suite for Phase 3 — MongoDB Connection & Session Persistence.
Tests:
- Async connection pool, index creation idempotency, and lifecycle
- SessionRepository CRUD operations and query filters
- Pipeline run logging
- FeedbackRepository operations
- GoldDatasetRepository and EvalRepository operations
- Dual-write strategy (MongoDB primary + JSONL fallback)
- Admin endpoints with MongoDB and fallback sources
"""
from __future__ import annotations
import os
import json
import asyncio
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from mongomock_motor import AsyncMongoMockClient
from fastapi.testclient import TestClient

from db.connection import connect_db, close_db, _ensure_indexes
from db.repositories import (
    SessionRepository,
    FeedbackRepository,
    GoldDatasetRepository,
    EvalRepository,
)
from db.models import GoldCaseModel, EvalRunModel
from schemas.output_schemas import PipelineSessionOutput, DiagnosisOutput
from app import app, persist_session, persist_feedback


@pytest_asyncio.fixture
async def mock_mongo_db():
    """Create an isolated in-memory AsyncMongoMock database instance."""
    client = AsyncMongoMockClient()
    db = client["medvoice_test"]
    await _ensure_indexes(db)
    yield db
    client.close()


@pytest.mark.asyncio
async def test_connection_lifecycle(monkeypatch):
    """Test connect_db and close_db lifecycle with custom client instance."""
    mock_client = AsyncMongoMockClient()
    db = await connect_db(client_instance=mock_client, db_name="medvoice_test_pool")
    assert db.name == "medvoice_test_pool"

    # Verify index creation does not crash
    await _ensure_indexes(db)
    await close_db()


@pytest.mark.asyncio
async def test_session_repository_save_and_get(mock_mongo_db):
    """Test saving a validated PipelineSessionOutput and fetching it by session_id."""
    repo = SessionRepository(mock_mongo_db)

    session_payload = PipelineSessionOutput(
        session_id="sess-test-101",
        status="ok",
        department="er",
        language="ar",
        mode="doctor",
        transcript="مريض يعاني من ألم في الصدر وضيق تنفس",
        extraction_sec=1.25,
        total_sec=2.15,
        diagnosis=DiagnosisOutput(
            final_diagnosis="Acute Coronary Syndrome",
            icd10_codes=["I20.0"],
            reasoning="Severe retrosternal chest pain with diaphoresis",
        ),
    )

    meta = {
        "ip": "127.0.0.1",
        "user_agent": "Mozilla/5.0",
        "clinician_name": "Dr. Ahmed",
        "sso_user": "ahmed_er",
    }

    doc_id = await repo.save_session(session_payload, meta=meta)
    assert doc_id is not None

    fetched = await repo.get_session("sess-test-101")
    assert fetched is not None
    assert fetched["session_id"] == "sess-test-101"
    assert fetched["department"] == "er"
    assert fetched["clinician_name"] == "Dr. Ahmed"
    assert fetched["diagnosis"]["final_diagnosis"] == "Acute Coronary Syndrome"
    assert fetched["diagnosis"]["icd10_codes"] == ["I20.0"]
    assert "created_at" in fetched


@pytest.mark.asyncio
async def test_session_repository_upsert_idempotency(mock_mongo_db):
    """Test that saving the same session_id updates rather than creating duplicate documents."""
    repo = SessionRepository(mock_mongo_db)

    session_v1 = PipelineSessionOutput(
        session_id="sess-upsert-1",
        status="ok",
        department="opd",
        language="en",
        mode="doctor",
        transcript="Patient with cough",
    )
    await repo.save_session(session_v1, meta={"clinician_name": "Dr. Smith"})

    session_v2 = PipelineSessionOutput(
        session_id="sess-upsert-1",
        status="ok",
        department="opd",
        language="en",
        mode="doctor",
        transcript="Patient with cough and fever",
    )
    await repo.save_session(session_v2, meta={"clinician_name": "Dr. Smith Updated"})

    sessions = await repo.list_sessions(limit=10)
    assert len(sessions) == 1
    assert sessions[0]["transcript"] == "Patient with cough and fever"
    assert sessions[0]["clinician_name"] == "Dr. Smith Updated"


@pytest.mark.asyncio
async def test_session_repository_filtering(mock_mongo_db):
    """Test filtering sessions by department and clinician_name."""
    repo = SessionRepository(mock_mongo_db)

    for i in range(3):
        sess = PipelineSessionOutput(
            session_id=f"sess-er-{i}",
            status="ok",
            department="er",
            language="ar",
            mode="conversation",
        )
        await repo.save_session(sess, meta={"clinician_name": "Dr. Mona"})

    for i in range(2):
        sess = PipelineSessionOutput(
            session_id=f"sess-dental-{i}",
            status="ok",
            department="dental",
            language="en",
            mode="doctor",
        )
        await repo.save_session(sess, meta={"clinician_name": "Dr. Tarek"})

    all_sess = await repo.list_sessions(limit=10)
    assert len(all_sess) == 5

    er_sess = await repo.list_sessions(limit=10, department="er")
    assert len(er_sess) == 3

    dental_sess = await repo.list_sessions(limit=10, department="dental")
    assert len(dental_sess) == 2

    mona_sess = await repo.list_sessions(limit=10, clinician_name="Dr. Mona")
    assert len(mona_sess) == 3


@pytest.mark.asyncio
async def test_pipeline_runs_logging(mock_mongo_db):
    """Test saving and retrieving granular pipeline run steps."""
    repo = SessionRepository(mock_mongo_db)

    run_id1 = await repo.save_pipeline_step(
        session_id="sess-run-1",
        step="extraction",
        input_data={"transcript": "Chest pain"},
        output_data={"chief_complaint": "Chest pain"},
        timings={"latency_sec": 0.8},
        usage={"prompt_tokens": 120, "completion_tokens": 45},
    )
    assert run_id1 is not None

    run_id2 = await repo.save_pipeline_step(
        session_id="sess-run-1",
        step="diagnosis",
        input_data={"chief_complaint": "Chest pain"},
        output_data={"diagnosis": "Angina"},
        timings={"latency_sec": 1.1},
        usage={"prompt_tokens": 200, "completion_tokens": 60},
    )
    assert run_id2 is not None

    runs = await repo.list_pipeline_runs("sess-run-1")
    assert len(runs) == 2
    assert runs[0]["step_name"] == "extraction"
    assert runs[1]["step_name"] == "diagnosis"
    assert runs[0]["timings"]["latency_sec"] == 0.8


@pytest.mark.asyncio
async def test_feedback_repository(mock_mongo_db):
    """Test saving feedback and querying recent entries."""
    repo = FeedbackRepository(mock_mongo_db)

    feed_id = await repo.save({
        "session_id": "sess-feed-1",
        "rating": "thumbs_up",
        "text": "Great clinical summary and exact ICD codes",
        "name": "Dr. Sarah",
        "department": "er",
        "language": "ar",
    })
    assert feed_id is not None

    entries = await repo.list_recent(limit=10)
    assert len(entries) == 1
    assert entries[0]["session_id"] == "sess-feed-1"
    assert entries[0]["rating"] == "thumbs_up"
    assert entries[0]["department"] == "er"
    assert "created_at" in entries[0]


@pytest.mark.asyncio
async def test_gold_dataset_and_eval_repositories(mock_mongo_db):
    """Test GoldDatasetRepository and EvalRepository for Phase 3 document schema compliance."""
    gold_repo = GoldDatasetRepository(mock_mongo_db)
    eval_repo = EvalRepository(mock_mongo_db)

    case = GoldCaseModel(
        case_id="gold-cardio-01",
        department="er",
        specialty="Cardiology",
        audio_language="ar",
        transcript="ألم حاد في الصدر يمتد للذراع الأيسر",
        expected_extraction={"chief_complaint": "Chest pain radiating to left arm"},
        expected_diagnosis={"final_diagnosis": "Myocardial Infarction"},
    )
    c_id = await gold_repo.save_case(case)
    assert c_id is not None

    retrieved_case = await gold_repo.get_case("gold-cardio-01")
    assert retrieved_case is not None
    assert retrieved_case["specialty"] == "Cardiology"

    eval_run = EvalRunModel(
        eval_id="eval-run-001",
        case_id="gold-cardio-01",
        session_id="sess-eval-1",
        model_id="gpt-5.6-luna",
        actual_extraction={"chief_complaint": "Chest pain radiating to left arm"},
        actual_diagnosis={"final_diagnosis": "Myocardial Infarction"},
        scores={"extraction_f1": 0.95, "diagnosis_match": 1.0},
    )
    e_id = await eval_repo.save_eval(eval_run)
    assert e_id is not None

    evals = await eval_repo.list_by_case("gold-cardio-01")
    assert len(evals) == 1
    assert evals[0]["scores"]["extraction_f1"] == 0.95


@pytest.mark.asyncio
async def test_persist_session_dual_write(mock_mongo_db, tmp_path, monkeypatch):
    """Test persist_session writes both to MongoDB and JSONL fallback seamlessly."""
    repo = SessionRepository(mock_mongo_db)

    # Temporary log file
    test_usage_log = tmp_path / "test_usage.jsonl"
    import app as app_module
    monkeypatch.setattr(app_module, "USAGE_LOG", test_usage_log)

    log_entry = {
        "session_id": "sess-dual-write-1",
        "department": "er",
        "language": "en",
        "mode": "doctor",
        "status": "ok",
        "transcript": "Headache and vomiting",
        "clinician_name": "Dr. Kareem",
        "total_sec": 1.5,
        "extraction_sec": 0.9,
    }

    await persist_session(log_entry, repo)

    # 1. Verify in MongoDB
    mongo_doc = await repo.get_session("sess-dual-write-1")
    assert mongo_doc is not None
    assert mongo_doc["clinician_name"] == "Dr. Kareem"
    assert mongo_doc["transcript"] == "Headache and vomiting"

    # 2. Verify in JSONL fallback
    assert test_usage_log.exists()
    lines = test_usage_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    saved_json = json.loads(lines[0])
    assert saved_json["session_id"] == "sess-dual-write-1"


@pytest.mark.asyncio
async def test_persist_feedback_dual_write(mock_mongo_db, tmp_path, monkeypatch):
    """Test persist_feedback writes to both MongoDB and JSONL."""
    repo = FeedbackRepository(mock_mongo_db)

    test_feedback_log = tmp_path / "test_feedback.jsonl"
    import app as app_module
    monkeypatch.setattr(app_module, "FEEDBACK_LOG", test_feedback_log)

    entry = {
        "session_id": "sess-fb-1",
        "rating": "thumbs_down",
        "text": "Missed allergy info",
        "department": "opd",
        "language": "ar",
    }

    await persist_feedback(entry, repo)

    # 1. MongoDB
    entries = await repo.list_recent(limit=5)
    assert len(entries) == 1
    assert entries[0]["text"] == "Missed allergy info"

    # 2. JSONL
    assert test_feedback_log.exists()
    lines = test_feedback_log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["rating"] == "thumbs_down"


def test_admin_routes_with_mongodb(mock_mongo_db):
    """Test /admin/usage and /admin/feedback read from MongoDB when connected."""
    app.state.db = mock_mongo_db
    app.state.sessions = SessionRepository(mock_mongo_db)
    app.state.feedback_repo = FeedbackRepository(mock_mongo_db)

    # Pre-populate mock MongoDB
    asyncio.run(app.state.sessions.save_session(
        PipelineSessionOutput(
            session_id="sess-admin-mongo",
            department="er",
            language="ar",
            mode="doctor",
            transcript="Pain in abdomen",
        ),
        meta={"clinician_name": "Dr. Admin"},
    ))

    asyncio.run(app.state.feedback_repo.save({
        "session_id": "sess-admin-mongo",
        "rating": "thumbs_up",
        "text": "Accurate differential diagnosis",
    }))

    client = TestClient(app)
    auth = ("DohaMahmoud", "1811")

    res_usage = client.get("/admin/usage", auth=auth)
    assert res_usage.status_code == 200
    data_usage = res_usage.json()
    assert data_usage["source"] == "mongodb"
    assert len(data_usage["entries"]) >= 1
    assert data_usage["entries"][0]["session_id"] == "sess-admin-mongo"

    res_fb = client.get("/admin/feedback", auth=auth)
    assert res_fb.status_code == 200
    data_fb = res_fb.json()
    assert data_fb["source"] == "mongodb"
    assert len(data_fb["entries"]) >= 1
    assert data_fb["entries"][0]["rating"] == "thumbs_up"
