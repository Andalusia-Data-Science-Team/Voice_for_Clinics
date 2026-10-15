import os
import time
import json
import logging
import asyncio
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from sse_starlette.sse import EventSourceResponse
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from core.config import Config
from model.speech_service import SpeechService
from model.llm_service import LLMService
from model.diagnosis import constrain_diagnosis_to_registry
from model.icd10_registry import get_registry
from model.question_generator import QuestionGenerator
from core.model_registry import get_model, registry_for_ui
from model.specialty_terms import correct_specialty_terms, SPECIALTIES
from utils.red_flags import detect_red_flags
from model.patient_history import (
    get_patient_visits,
    format_history_for_prompt,
    check_history_usage,
    BusinessUnit as _PatientBU,   # aliased to avoid clash with schemas.BusinessUnit
    PatientHistoryError,
)
from llm.runtime import LLMRuntime, get_llm_runtime

# ── Phase 2: schema + middleware imports ─────────────────────────────────────
from schemas.input_schemas import (
    AnalyzeInput,
    FeedbackInput,
    Language,
    Mode,
    Department,
    BusinessUnit,
    RefineFieldInput,
)
from validation.middleware import (
    validate_analyze_input,
    validate_refine_field_input,
    validate_feedback_input,
    validate_extraction_output,
    validate_diagnosis_output,
)
from schemas.output_schemas import PipelineSessionOutput
from schemas.gold_schemas import GoldCase
from db import (
    connect_db,
    close_db,
    get_db,
    SessionRepository,
    FeedbackRepository,
    GoldDatasetRepository,
    EvalRepository,
)
from evaluation import EvaluationEngine


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio Processing"])
patient_router = APIRouter(prefix="/patient", tags=["Patient History"])


def _sse_json(payload: object) -> str:
    """Serialize SSE data as compact, standards-compliant UTF-8 JSON."""
    # Arabic text is much smaller as UTF-8 than as ``\\uXXXX`` escapes, and
    # strict JSON avoids sending Python-only NaN/Infinity values to JSON.parse.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def get_current_request(request: Request) -> Request:
    """Provide the current Starlette request through FastAPI dependency injection."""
    return request


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and close shared provider adapters and MongoDB connection per worker."""
    from providers.factory import build_llm_provider, build_stt_provider

    # Initialize M2 providers
    app.state.llm_provider = build_llm_provider()
    app.state.stt_provider = build_stt_provider()
    # Backward compatibility alias for LLMRuntime consumers
    app.state.llm = getattr(app.state.llm_provider, "runtime", app.state.llm_provider)

    try:
        app.state.db = await connect_db()
        app.state.sessions = SessionRepository(app.state.db)
        app.state.feedback_repo = FeedbackRepository(app.state.db)
        logger.info("[Lifespan] MongoDB connection and repositories initialized successfully")
    except Exception as exc:
        app.state.db = None
        app.state.sessions = None
        app.state.feedback_repo = None
        logger.warning("[Lifespan] MongoDB initialization skipped/failed: %s (JSONL fallback active)", exc)

    try:
        yield
    finally:
        if hasattr(app.state.llm_provider, "close"):
            await app.state.llm_provider.close()
        elif hasattr(app.state.llm, "close"):
            await app.state.llm.close()
        await close_db()


app = FastAPI(title="Medical Voice API", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


@app.get("/")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon():
    """Return the application logo for browsers requesting a favicon."""
    return FileResponse(Path(__file__).parent / "static" / "andalusia.png")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1110",
        "https://medical-voice-assistant.ahbsdev.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Usage logging
# ═══════════════════════════════════════════════════════════════════════════════

LOGS_DIR  = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
USAGE_LOG = LOGS_DIR / "usage.jsonl"


def _get_client_ip(request: Request) -> str:
    """Return the real client IP, respecting X-Forwarded-For (ngrok / reverse proxy)."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_client_user(request: Request) -> Optional[str]:
    """
    Best-effort identification of who's using the app, beyond what the
    person typed into a name field themselves.

    A browser will NEVER expose the machine's logged-in OS/Windows username
    to a web page — that's a hard security boundary built into every
    browser, not something fixable from this app's code. The only way to
    get an authenticated username automatically is if something in front of
    this app authenticates the request and passes the result along, e.g. a
    reverse proxy doing Windows-integrated auth (Kerberos/NTLM) or SSO,
    which typically injects the authenticated identity as a header such as
    X-Forwarded-User. This checks for a few common ones, in priority order,
    and returns None if none are present — callers should fall back to a
    self-reported name (clinician_name / feedback's name field) in that case.
    """
    for header in ("x-forwarded-user", "x-remote-user", "x-auth-user", "remote-user"):
        val = request.headers.get(header)
        if val and val.strip():
            return val.strip()
    return None


def save_usage_log(entry: dict) -> None:
    try:
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("[UsageLog] Saved entry for %s", entry.get("ip"))
    except Exception as e:
        logger.warning("[UsageLog] Failed to write log: %s", e)


async def persist_session(log: dict, session_repo: Optional[SessionRepository] = None) -> None:
    """Write session to MongoDB with schema validation, always writing to JSONL fallback or on DB failure."""
    # Dual-write: Write to JSONL fallback in background thread
    await asyncio.to_thread(save_usage_log, log)

    if session_repo is not None:
        try:
            validated = PipelineSessionOutput.model_validate(log)
            await session_repo.save_session(
                validated,
                meta={
                    "ip": log.get("ip"),
                    "user_agent": log.get("user_agent"),
                    "sso_user": log.get("sso_user"),
                    "clinician_name": log.get("clinician_name"),
                    "patient_id": log.get("patient_id"),
                    "business_unit": log.get("business_unit"),
                    "history_check": log.get("history_check"),
                },
            )
            logger.info("[Persist] Session %s successfully persisted to MongoDB", log.get("session_id"))
        except Exception as exc:
            logger.error("[Persist] MongoDB session write failed: %s (JSONL write preserved)", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Doctor feedback — free-text feedback/complaint/suggestion + thumbs up/down,
# submitted from the small feedback card at the bottom of the page. Written
# both to its own JSONL log (same append-only pattern as usage.jsonl above,
# so it's easy to page through or pull into a spreadsheet later) AND to the
# regular app logger at INFO level with a "[Feedback]" tag, so it shows up
# directly in whatever log file/console this app's logging is already wired
# to, without needing to go find feedback.jsonl specifically.
# ═══════════════════════════════════════════════════════════════════════════════

FEEDBACK_LOG = LOGS_DIR / "feedback.jsonl"


def save_feedback_log(entry: dict) -> None:
    try:
        with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[Feedback] Failed to write feedback log: %s", e)


async def persist_feedback(entry: dict, feedback_repo: Optional[FeedbackRepository] = None) -> None:
    """Dual-write feedback entry to MongoDB and JSONL."""
    await asyncio.to_thread(save_feedback_log, entry)
    if feedback_repo is not None:
        try:
            await feedback_repo.save(entry)
            logger.info("[Persist] Feedback successfully persisted to MongoDB")
        except Exception as exc:
            logger.error("[Persist] MongoDB feedback write failed: %s (JSONL preserved)", exc)


@app.get("/models", summary="Fixed model per pipeline step")
async def list_models():
    return registry_for_ui()


@app.post("/feedback", summary="Submit doctor feedback (thumbs rating and/or free text)")
async def submit_feedback(
    request: Request,
    inp: FeedbackInput = Depends(validate_feedback_input),
):
    """Validated by FeedbackInput via validate_feedback_input dependency.
    Dual-writes to MongoDB and JSONL fallback.
    """
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "session_id": inp.session_id,
        "ip":         _get_client_ip(request),
        "name":       inp.name,
        "sso_user":   _get_client_user(request),
        "rating":     inp.rating,
        "text":       inp.text,
        "department": inp.department,
        "language":   inp.language,
    }
    logger.info(
        "[Feedback] session_id=%s name=%s rating=%s department=%s language=%s text=%r",
        entry["session_id"], entry["name"], entry["rating"], entry["department"], entry["language"], entry["text"],
    )
    feedback_repo: Optional[FeedbackRepository] = getattr(request.app.state, "feedback_repo", None)
    await persist_feedback(entry, feedback_repo)
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics dashboard — reads usage.jsonl + feedback.jsonl and serves them to
# a small static dashboard page (static/dashboard.html) for at-a-glance
# monitoring: session volume, department/language mix, error rate, and the
# feedback stream (thumbs + free text). Nothing here is computed/aggregated
# server-side beyond basic reading — the dashboard page does its own
# aggregation client-side from the raw entries, so there's one source of
# truth (the JSONL files) and no separate "stats" logic to keep in sync.
#
# ⚠ ACCESS CONTROL: this is genuinely sensitive — usage.jsonl includes full
# transcripts and extracted clinical note content, and feedback may include
# clinician names. If Config.DASHBOARD_USERNAME and Config.DASHBOARD_PASSWORD
# are both set, every route below requires that HTTP Basic Auth login. If
# either is unset, THE DASHBOARD IS OPEN TO ANYONE WHO KNOWS THE URL — that's
# a deliberate "works out of the box in dev" default, not a recommendation;
# set both before exposing this outside a trusted network.
# ═══════════════════════════════════════════════════════════════════════════════

_dashboard_security = HTTPBasic(auto_error=False)


def _require_dashboard_auth(credentials: Optional[HTTPBasicCredentials] = Depends(_dashboard_security)):
    dash_user = getattr(Config, "DASHBOARD_USERNAME", None)
    dash_pass = getattr(Config, "DASHBOARD_PASSWORD", None)
    if not dash_user or not dash_pass:
        return  # no credentials configured — dashboard runs unauthenticated, see warning above
    ok = bool(credentials) and secrets.compare_digest(credentials.username, dash_user) \
                            and secrets.compare_digest(credentials.password, dash_pass)
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})


def _read_jsonl(path: Path, limit: int) -> list:
    """Every entry in an append-only JSONL log, newest first, capped at limit.
    Malformed lines are skipped rather than failing the whole read."""
    if not path.exists():
        return []
    entries = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"[Dashboard] Failed to read {path}: {e}")
        return []
    entries.reverse()
    return entries[:limit]


@app.get("/admin/usage", summary="Recent /audio/analyze usage log entries, newest first")
async def admin_usage(
    request: Request,
    limit: int = 500,
    department: Optional[str] = None,
    _auth=Depends(_require_dashboard_auth),
):
    limit = max(1, min(limit, 5000))
    session_repo: Optional[SessionRepository] = getattr(request.app.state, "sessions", None)
    if session_repo is not None:
        try:
            docs = await session_repo.list_sessions(limit=limit, department=department)
            return {"entries": docs, "source": "mongodb"}
        except Exception as exc:
            logger.warning("[Dashboard] Failed to read sessions from MongoDB, falling back to JSONL: %s", exc)

    entries = _read_jsonl(USAGE_LOG, limit)
    if department:
        entries = [e for e in entries if e.get("department") == department]
    return {"entries": entries, "source": "jsonl"}


@app.get("/admin/feedback", summary="Recent doctor feedback entries, newest first")
async def admin_feedback(
    request: Request,
    limit: int = 500,
    department: Optional[str] = None,
    _auth=Depends(_require_dashboard_auth),
):
    limit = max(1, min(limit, 5000))
    feedback_repo: Optional[FeedbackRepository] = getattr(request.app.state, "feedback_repo", None)
    if feedback_repo is not None:
        try:
            docs = await feedback_repo.list_recent(limit=limit, department=department)
            return {"entries": docs, "source": "mongodb"}
        except Exception as exc:
            logger.warning("[Dashboard] Failed to read feedback from MongoDB, falling back to JSONL: %s", exc)

    entries = _read_jsonl(FEEDBACK_LOG, limit)
    if department:
        entries = [e for e in entries if e.get("department") == department]
    return {"entries": entries, "source": "jsonl"}


@app.get("/dashboard", summary="Analytics dashboard — usage + feedback, for internal review")
async def serve_dashboard(_auth=Depends(_require_dashboard_auth)):
    return FileResponse(Path(__file__).parent / "static" / "dashboard.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Observability — /health, /health/live, /admin/status
# ═══════════════════════════════════════════════════════════════════════════════

_APP_START_TIME: float = time.monotonic()


async def _check_mongo(request: Request) -> dict:
    """Ping MongoDB and return a structured health sub-check dict."""
    from db.connection import get_mongo_status
    try:
        status = await asyncio.wait_for(get_mongo_status(), timeout=3.0)
        return {
            "component": "mongodb",
            "status": "ok" if status.get("connected") else "degraded",
            "latency_ms": status.get("latency_ms"),
            "database": status.get("database"),
            "pool": {
                "max": status.get("max_pool_size"),
                "min": status.get("min_pool_size"),
            },
            "detail": status.get("error"),
        }
    except asyncio.TimeoutError:
        return {"component": "mongodb", "status": "timeout", "detail": "ping timed out after 3 s"}
    except Exception as exc:
        return {"component": "mongodb", "status": "error", "detail": str(exc)}


async def _check_llm(request: Request) -> dict:
    """Probe LLM reachability via the active LLMProvider abstraction.

    Calls provider.health_check() which performs a lightweight probe without tokens.
    """
    from providers.factory import get_llm
    provider = getattr(request.app.state, "llm_provider", None) or get_llm()
    try:
        result = await provider.health_check()
        return {
            "component": "llm",
            "provider": getattr(provider, "provider_name", os.getenv("LLM_PROVIDER", "foundry")),
            "status": "ok" if result.ok else "degraded",
            "latency_ms": result.latency_ms,
            "detail": result.detail,
        }
    except Exception as exc:
        return {
            "component": "llm",
            "provider": getattr(provider, "provider_name", os.getenv("LLM_PROVIDER", "foundry")),
            "status": "error",
            "detail": str(exc),
        }


def _check_buffer(request: Request) -> dict:
    """Return the current write-behind buffer state (synchronous)."""
    session_repo: Optional[SessionRepository] = getattr(request.app.state, "sessions", None)
    if session_repo is None:
        return {"component": "write_buffer", "status": "disabled", "buffered": 0}
    buffered = len(session_repo._buffer) if hasattr(session_repo, "_buffer") else 0
    flusher_alive = (
        session_repo._flush_task is not None
        and not session_repo._flush_task.done()
        if hasattr(session_repo, "_flush_task")
        else False
    )
    return {
        "component": "write_buffer",
        "status": "ok" if not session_repo._enable_buffer or flusher_alive else "degraded",
        "buffered": buffered,
        "buffer_enabled": getattr(session_repo, "_enable_buffer", False),
        "flusher_alive": flusher_alive,
    }


def _check_disk() -> dict:
    """Spot-check that the logs directory is writable."""
    try:
        probe = LOGS_DIR / ".healthcheck"
        probe.write_text("ok")
        probe.unlink()
        return {"component": "disk", "status": "ok", "logs_dir": str(LOGS_DIR)}
    except Exception as exc:
        return {"component": "disk", "status": "error", "detail": str(exc)}


@app.get(
    "/health",
    summary="Readiness probe — checks MongoDB, LLM, buffer, and disk",
    tags=["Observability"],
    response_model=None,
)
async def health_check(request: Request):
    """Concurrent health check across all critical sub-systems.

    Returns HTTP 200 when every sub-system is ``ok``.
    Returns HTTP 503 when any *critical* sub-system (MongoDB or LLM) is degraded.
    Non-critical failures (buffer, disk) downgrade ``overall`` to ``degraded``
    but still return HTTP 200 so the process stays in the load-balancer rotation.
    """
    uptime_s = round(time.monotonic() - _APP_START_TIME, 1)

    # Run all async checks concurrently — no sub-system can block another
    mongo_result, llm_result = await asyncio.gather(
        _check_mongo(request),
        _check_llm(request),
        return_exceptions=False,
    )
    buffer_result = _check_buffer(request)
    disk_result = _check_disk()

    checks = [mongo_result, llm_result, buffer_result, disk_result]

    # Determine overall status
    critical_statuses = {mongo_result["status"], llm_result["status"]}
    if "error" in critical_statuses or "unreachable" in critical_statuses or "timeout" in critical_statuses:
        overall = "unhealthy"
        http_status = 503
    elif any(c["status"] not in ("ok",) for c in checks):
        overall = "degraded"
        http_status = 200
    else:
        overall = "ok"
        http_status = 200

    body = {
        "overall": overall,
        "uptime_seconds": uptime_s,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }

    return Response(
        content=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        status_code=http_status,
        media_type="application/json",
    )


@app.get(
    "/health/live",
    summary="Liveness probe — fast in-process check (no I/O)",
    tags=["Observability"],
    response_model=None,
)
async def health_live():
    """Ultra-lightweight liveness probe for Kubernetes/Docker health checks.

    Does *not* probe MongoDB or LLM — only confirms the process is alive and
    the event loop is responding. Returns HTTP 200 always.
    """
    return Response(
        content=json.dumps({
            "status": "alive",
            "uptime_seconds": round(time.monotonic() - _APP_START_TIME, 1),
        }, separators=(",", ":")),
        status_code=200,
        media_type="application/json",
    )


@app.get(
    "/admin/status",
    summary="Detailed system diagnostics (admin-only)",
    tags=["Admin"],
)
async def admin_status(request: Request, _auth=Depends(_require_dashboard_auth)):
    """Extended diagnostics combining health sub-checks with pool internals,
    buffer metrics, and JSONL fallback sizes. Requires dashboard credentials.
    """
    uptime_s = round(time.monotonic() - _APP_START_TIME, 1)

    mongo_result, llm_result = await asyncio.gather(
        _check_mongo(request),
        _check_llm(request),
    )
    buffer_result = _check_buffer(request)
    disk_result = _check_disk()

    # JSONL fallback file sizes for observability
    jsonl_sizes = {}
    for name, path in [("usage", USAGE_LOG), ("feedback", FEEDBACK_LOG)]:
        try:
            jsonl_sizes[name] = {"path": str(path), "size_bytes": path.stat().st_size if path.exists() else 0}
        except Exception:
            jsonl_sizes[name] = {"path": str(path), "size_bytes": None}

    overall_statuses = {mongo_result["status"], llm_result["status"]}
    if "error" in overall_statuses or "unreachable" in overall_statuses or "timeout" in overall_statuses:
        overall = "unhealthy"
    elif any(c["status"] != "ok" for c in [mongo_result, llm_result, buffer_result, disk_result]):
        overall = "degraded"
    else:
        overall = "ok"

    return {
        "overall": overall,
        "uptime_seconds": uptime_s,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": [mongo_result, llm_result, buffer_result, disk_result],
        "jsonl_fallback": jsonl_sizes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Endpoints — Gold cases management, automated evaluation, & reports
# ═══════════════════════════════════════════════════════════════════════════════

eval_router = APIRouter(prefix="/eval", tags=["Evaluation"])


def _get_eval_repos(request: Request):
    """Retrieve initialized database repositories or raise HTTPException 503."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection is not available")
    gold_repo = GoldDatasetRepository(db)
    eval_repo = EvalRepository(db)
    session_repo = SessionRepository(db)
    return gold_repo, eval_repo, session_repo


@eval_router.post("/gold-cases", summary="Upload a gold case")
async def create_gold_case(case: GoldCase, request: Request):
    gold_repo, _, _ = _get_eval_repos(request)
    case_id = await gold_repo.upsert_case(case)
    return {"case_id": case_id, "status": "created"}


@eval_router.get("/gold-cases", summary="List gold cases")
async def list_gold_cases(
    request: Request,
    department: Optional[str] = None,
    limit: int = 100,
):
    gold_repo, _, _ = _get_eval_repos(request)
    cases = await gold_repo.list_cases(department=department, limit=limit)
    return {"cases": cases, "count": len(cases)}


@eval_router.post("/gold-cases/import-xlsx", summary="Import gold cases from Excel")
async def import_xlsx(request: Request, file_path: Optional[str] = None):
    gold_repo, _, _ = _get_eval_repos(request)
    target_path = file_path or getattr(Config, "XLSX_PATH", None)
    if not target_path or not Path(target_path).exists():
        raise HTTPException(status_code=404, detail=f"Gold dataset Excel file not found at: {target_path}")

    count = await gold_repo.import_from_xlsx(str(target_path))
    return {"imported": count, "file": str(target_path)}


@eval_router.post("/run/{case_id}", summary="Evaluate a session against a gold case")
async def run_evaluation(case_id: str, session_id: str, request: Request):
    gold_repo, eval_repo, session_repo = _get_eval_repos(request)
    engine = EvaluationEngine(
        gold_repo=gold_repo,
        eval_repo=eval_repo,
        session_repo=session_repo,
    )
    try:
        result = await engine.evaluate_session(session_id=session_id, case_id=case_id)
        return result.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@eval_router.get("/report", summary="Evaluation summary report")
async def eval_report(request: Request, department: Optional[str] = None):
    _, eval_repo, _ = _get_eval_repos(request)
    return await eval_repo.aggregate_report(department=department)


@eval_router.get("/runs", summary="Recent evaluation runs")
async def list_eval_runs(
    request: Request,
    department: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 100,
):
    _, eval_repo, _ = _get_eval_repos(request)
    runs = await eval_repo.list_recent_runs(department=department, model_id=model_id, limit=limit)
    return {"runs": runs, "count": len(runs)}



# ═══════════════════════════════════════════════════════════════════════════════
# Enums — now imported from schemas.input_schemas (Phase 2)
# Language, Mode, Department are imported at the top of this file.
# The dynamic Specialty enum is retained here only for any legacy callers
# outside the main endpoints.
# ═══════════════════════════════════════════════════════════════════════════════

# Built from model.specialty_terms.SPECIALTIES so the two never drift apart.
from enum import Enum as _Enum
Specialty = _Enum("Specialty", {s.replace(" ", "_").replace("&", "and"): s for s in SPECIALTIES}, type=str)


# ═══════════════════════════════════════════════════════════════════════════════
# GET /audio/tts-disclaimer — Munsit TTS recording-consent line (Lina, Fusha)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/audio/tts-disclaimer",
    summary="Recording-consent disclaimer audio, spoken by Lina (Fusha/MSA) via Munsit TTS",
)
async def tts_disclaimer():
    """
    Returns a WAV file of the fixed Arabic recording-consent line, synthesized
    via Munsit TTS (voice: Lina, dialect: fusha, model: faseeh-v1-preview).
    The text never changes, so SpeechService caches the rendered audio
    in-process after the first successful call.

    Played once, client-side, at the very start of a new recording session —
    never again on pause/resume within that same session.
    """
    try:
        audio_bytes = await SpeechService.get_recording_disclaimer_audio(Config.MUNSIT_API_KEY)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error("[TTS] Disclaimer synthesis failed: %s", e)
        return Response(
            content=json.dumps({"detail": str(e)}),
            media_type="application/json",
            status_code=502,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GET /patient/{patient_code}/history?business_unit=AHJ|AMH|ANC — "Show Patient
# Previous Visits Over the Last Year" button, and the same lookup that feeds the
# context sent to the LLM when the doctor opts in to "Use patient history?" on
# /audio/analyze.
#
# Each business unit has its own HCMS database, so business_unit is required.
# A failed lookup (DB unreachable, bad credentials, ...) is a 503 — deliberately
# different from a patient with no history, which is a normal 200 with
# visit_count 0.
# ═══════════════════════════════════════════════════════════════════════════════

@patient_router.get(
    "/{patient_code}/history",
    summary="Visit history for a PatientCode in one business unit, newest first",
)
async def patient_history(patient_code: str, business_unit: BusinessUnit):
    try:
        visits = await asyncio.to_thread(get_patient_visits, patient_code, business_unit)
    except PatientHistoryError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "patient_code":  patient_code,
        "business_unit": business_unit.value,
        "visit_count":   len(visits),
        "visits":        visits,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — live real-time transcription
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    Real-time transcription — routed by language:
      language == "en" → Voxtral   (unchanged from before)
      language == "ar" → Munsit    (dialect-aware Arabic, incl. Khaliji/Gulf)

    Protocol (client → server):
      1. First text message: JSON  {"sampleRate": <int>, "language": "en"|"ar", "hotwords": "..."}
         "language" defaults to "en" if omitted, for backward compatibility.
      2. Binary messages: float32 little-endian PCM mono chunks.
         The client may simply stop sending these for a while (pause) without
         closing the socket — Munsit's server-side KeepAlive loop (see
         speech_service.py) keeps the session alive through the gap, and
         resuming binary messages picks the same transcript back up.
      3. Final text message: JSON  {"cmd": "done"}

    Protocol (server → client):
      {"type": "delta",   "text": "..."}   — committed text, append to transcript
      {"type": "interim", "text": "..."}   — provisional text, replace preview (Munsit only)
      {"type": "done"}
      {"type": "error",  "detail": "..."}
    """
    await websocket.accept()
    logger.info("[WS] Client connected — %s", websocket.client.host)

    input_sample_rate = 16000
    language = "ar"
    hotwords = None
    try:
        init_text = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        init_data = json.loads(init_text)
        input_sample_rate = int(init_data.get("sampleRate", 16000))
        language = str(init_data.get("language", "ar")).lower()
        hotwords = init_data.get("hotwords") or None
        logger.info("[WS] Browser sample rate: %d Hz, language: %s", input_sample_rate, language)
    except asyncio.TimeoutError:
        logger.warning("[WS] No init message within 5 s, assuming ar/16000 Hz")
    except Exception as e:
        logger.warning("[WS] Bad init message: %s — assuming ar/16000 Hz", e)

    pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)

    async def receive_loop() -> None:
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    logger.info("[WS] Client disconnected")
                    break
                if msg.get("bytes"):
                    await pcm_queue.put(msg["bytes"])
                elif msg.get("text"):
                    try:
                        cmd = json.loads(msg["text"])
                        if cmd.get("cmd") == "done":
                            logger.info("[WS] Received done signal")
                            break
                    except Exception:
                        pass
        except WebSocketDisconnect:
            logger.info("[WS] WebSocketDisconnect in receive_loop")
        except Exception as e:
            logger.warning("[WS] receive_loop error: %s", e)
        finally:
            await pcm_queue.put(None)

    async def pcm_iter():
        while True:
            chunk = await pcm_queue.get()
            if chunk is None:
                return
            yield chunk

    receive_task = asyncio.create_task(receive_loop())

    try:
        if language == "ar":
            # ── Arabic → Munsit ──
            async for event in SpeechService.transcribe_live_stream_ar(
                raw_pcm_iter=pcm_iter(),
                api_key=Config.MUNSIT_API_KEY,
                input_sample_rate=input_sample_rate,
                code_switching=True,
                hotwords=hotwords,
            ):
                etype = event.get("type")
                try:
                    if etype == "interim":
                        await websocket.send_json({"type": "interim", "text": event.get("text", "")})
                    elif etype == "final":
                        # Munsit sends one committed segment per turn — join with a
                        # leading space so consecutive turns don't run together.
                        text = event.get("text", "")
                        if text:
                            await websocket.send_json({"type": "delta", "text": text + " "})
                    elif etype == "error":
                        await websocket.send_json({"type": "error", "detail": event.get("detail", "Munsit error")})
                        if not event.get("recoverable", False):
                            break
                    elif etype == "done":
                        break
                    # "utterance_end" — no client action needed today.
                except (WebSocketDisconnect, Exception):
                    logger.info("[WS] Client gone mid-stream (ar)")
                    break
        else:
            # ── English → Voxtral (unchanged) ──
            async for delta in SpeechService.transcribe_live_stream(
                raw_pcm_iter=pcm_iter(),
                api_key=Config.MISTRAL_API_KEY,
                input_sample_rate=input_sample_rate,
            ):
                try:
                    await websocket.send_json({"type": "delta", "text": delta})
                except (WebSocketDisconnect, Exception):
                    logger.info("[WS] Client gone mid-stream (en)")
                    break

        try:
            await websocket.send_json({"type": "done"})
            logger.info("[WS] Transcription complete")
        except Exception:
            pass

    except Exception as e:
        logger.error("[WS] Transcription error: %s", e)
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass

    finally:
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("[WS] Session closed")


# ═══════════════════════════════════════════════════════════════════════════════
# POST /audio/refine-field — refine ONE spoken answer for a single missing
# clinical-note field (used by the "answer by voice" mic buttons next to a
# missing field / a "Questions for the doctor" item). Deliberately NOT part of
# the SSE /audio/analyze pipeline: this is a single short utterance, not a
# full visit dictation, so it gets a quick, single-shot JSON response instead
# of a streamed one. Runs it through the exact same RefineText module the
# full pipeline uses (deterministic specialty-term pass + light LLM cleanup),
# then — same as the main pipeline — translates Arabic answers to English,
# since every clinical-note field is always written in English regardless of
# which language the doctor spoke in.
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/refine-field",
    summary="Refine one short spoken answer for a single clinical-note field",
)
async def refine_field_answer(
    runtime: LLMRuntime = Depends(get_llm_runtime),
    inp: RefineFieldInput = Depends(validate_refine_field_input),
):
    """Validated by RefineFieldInput via validate_refine_field_input dependency.
    Inline blank-text guard removed — schema enforces it before the body runs.
    """
    text            = inp.text
    language        = inp.language
    specialty_value = inp.specialty  # already Optional[str], whitespace-stripped

    try:
        refine_fn = (
            LLMService.refine_ar_transcription_async
            if language == Language.arabic
            else LLMService.refine_en_transcription_async
        )
        refined_text = await refine_fn(text, runtime, is_conversation=False, specialty=specialty_value)
    except Exception as exc:
        logger.error("[RefineField] Failed to refine field answer: %s", exc)
        # Fall back to the raw (but specialty-corrected) transcript rather than
        # blocking the doctor from filling the field at all.
        refined_text = correct_specialty_terms(text, specialty_value)

    # Clinical-note fields are always English — translate if the answer was
    # spoken (and therefore refined) in Arabic, same as the main /audio/analyze
    # pipeline does for the full transcript.
    if language == Language.arabic:
        try:
            refined_text = await LLMService.translate_to_eng_async(
                refined_text, runtime, is_conversation=False
            )
        except Exception as exc:
            logger.error("[RefineField] Failed to translate field answer to English: %s", exc)
            # Leave the doctor with the refined Arabic text rather than nothing —
            # still better than losing the answer outright, and it's editable.

    return {"refined_text": refined_text}


# ═══════════════════════════════════════════════════════════════════════════════
# POST /audio/analyze — post-transcription pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/analyze",
    summary="Run refine → translate → extract → questions on a provided transcript",
)
async def analyze_transcript(
    request: Request,
    runtime: LLMRuntime = Depends(get_llm_runtime),
    inp: AnalyzeInput   = Depends(validate_analyze_input),
):
    """Validated by AnalyzeInput via validate_analyze_input dependency.

    All Form parameters are now pre-validated and pre-sanitized by the schema:
    - text is Unicode-normalized and whitespace-collapsed
    - use_history cross-field constraint (requires patient_id + business_unit)
      is enforced before this body runs — the inline guard below is removed
    - All string fields are stripped of leading/trailing whitespace
    """
    # ── Unpack validated fields (all pre-sanitized) ───────────────────────────
    text              = inp.text
    language          = inp.language
    mode              = inp.mode
    department_value  = inp.department.value
    specialty_value   = inp.specialty         # Optional[str], already clean
    use_history       = inp.use_history
    patient_id        = inp.patient_id
    session_id        = inp.session_id
    clinician_name    = inp.clinician_name
    transcription_sec = inp.transcription_sec

    # Convert to model.patient_history.BusinessUnit for the DB helper
    # (both are str-enums with identical values; explicit conversion is safer)
    _bu = _PatientBU(inp.business_unit.value) if inp.business_unit else None

    pipeline_start  = time.perf_counter()
    is_arabic       = language == Language.arabic
    is_conversation = mode == Mode.conversation
    skip_refinement = mode == Mode.doctor   # Doctor mode: skip LLM refinement step

    try:
        extraction_spec = get_model("extraction", None)
        reasoning_spec  = get_model("reasoning",  None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Patient history lookup — only when the doctor opted in.
    # The cross-field invariant (use_history ⟹ patient_id + business_unit)
    # is now enforced by AnalyzeInput, so the inline guard is removed here.
    #
    # A lookup that FAILS (DB unreachable, bad credentials, ...) does not abort
    # the analysis: the note is still generated, just without history context,
    # and history_error is passed to the frontend so the doctor is told history
    # was requested but NOT applied.
    patient_history_visits: list = []
    patient_history_text   = ""
    history_error: Optional[str] = None
    if use_history:
        try:
            patient_history_visits = await asyncio.to_thread(get_patient_visits, patient_id, _bu)
            patient_history_text   = format_history_for_prompt(patient_history_visits)
            logger.info(
                "[Analyze] use_history=true for patient_code=%r (%s) — found %d prior visit(s)",
                patient_id, inp.business_unit.value, len(patient_history_visits),
            )
        except PatientHistoryError as exc:
            history_error = str(exc)
            logger.error(
                "[Analyze] History lookup failed for patient_code=%r (%s) — continuing without history: %s",
                patient_id, inp.business_unit.value, exc,
            )

    # Deterministic fuzzy-matching pass — corrects near-miss specialty terms
    # in the raw transcript before anything else. Runs regardless of mode, so
    # Doctor-mode transcripts (which skip LLM refinement entirely) still get
    # it; Conversation-mode transcripts get it again implicitly via the LLM
    # refinement prompt's specialty context below.
    text = correct_specialty_terms(text, specialty_value)

    async def event_stream():
        # Accumulated outputs — populated as each step completes.
        # clinician_name is already stripped by AnalyzeInput (Phase 2).
        log: dict = {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "session_id":         session_id,
            "ip":                 _get_client_ip(request),
            "user_agent":         request.headers.get("user-agent", "unknown"),
            "clinician_name":     clinician_name,   # pre-sanitized by AnalyzeInput
            "sso_user":           _get_client_user(request),
            "language":           language.value,
            "mode":               mode.value,
            "department":         department_value,
            "specialty":          specialty_value,
            "patient_id":         patient_id,
            "business_unit":      inp.business_unit.value if inp.business_unit else None,
            "use_history":        use_history,
            "history_visit_count": len(patient_history_visits),
            "history_error":      history_error,
            "status":             "ok",
            "total_sec":          None,
            "transcription_sec":  transcription_sec,  # client-measured, not computed by this endpoint
            "refinement_sec":     None,
            "translation_sec":    None,
            "extraction_sec":     None,
            "question_generation_sec": None,
            "transcript":         text,
            "refined_text":       None,
            "refinement_skipped": skip_refinement,
            "translated_text":    None,
            "extracted_features": None,
            "routing_signal": None,
            "questions":          None,
            "extraction_model":   extraction_spec.id,
            "reasoning_model":    reasoning_spec.id,
            "diagnosis_sec":      None,
            "diagnosis":          None,
            "extraction_usage":   None,   # latency/tokens/cost of the LLM call
            "diagnosis_usage":    None,
        }
        logger.info(
            "[Analyze] session start — clinician_name=%r sso_user=%r department=%s language=%s mode=%s",
            log["clinician_name"], log["sso_user"], department_value, language.value, mode.value,
        )
        parallel_tasks = []

        try:
            # ── 1. Refine ─────────────────────────────────────────────────
            t0 = time.perf_counter()

            if skip_refinement:
                # Doctor mode: skip the refinement LLM call entirely and pass the
                # (possibly user-edited) transcript straight through to translate/extract.
                refined_text = text
                log["refined_text"] = refined_text
                log["refinement_sec"] = 0.0  # skipped entirely in Doctor mode — not "instant", just not run
                yield {"event": "refinement_skipped", "data": _sse_json({
                    "refined_text": refined_text,
                })}
            else:
                refined_parts: list[str] = []
                try:
                    refine_stream = (
                        LLMService.refine_ar_transcription_stream_async(text, runtime, is_conversation=is_conversation, specialty=specialty_value)
                        if is_arabic else
                        LLMService.refine_en_transcription_stream_async(text, runtime, is_conversation=is_conversation, specialty=specialty_value)
                    )
                    async for delta in refine_stream:
                        refined_parts.append(delta)
                        yield {"event": "refinement_delta", "data": _sse_json({"delta": delta})}
                except Exception as exc:
                    log["status"] = "error"
                    yield {"event": "error", "data": _sse_json({"step": "refinement", "detail": str(exc)})}
                    return

                refined_text      = "".join(refined_parts)
                refinement_sec    = round(time.perf_counter() - t0, 3)
                log["refined_text"]   = refined_text
                log["refinement_sec"] = refinement_sec
                yield {"event": "refinement_done", "data": _sse_json({
                    "refined_text":   refined_text,
                    "refinement_sec": refinement_sec,
                })}

            # ── 2. Translate (Arabic only) ────────────────────────────────
            if is_arabic:
                t0 = time.perf_counter()
                translated_parts: list[str] = []
                try:
                    async for delta in LLMService.translate_to_eng_stream_async(
                        refined_text, runtime, is_conversation=is_conversation
                    ):
                        translated_parts.append(delta)
                        yield {"event": "translation_delta", "data": _sse_json({"delta": delta})}
                except Exception as exc:
                    log["status"] = "error"
                    yield {"event": "error", "data": _sse_json({"step": "translation", "detail": str(exc)})}
                    return

                final_text             = "".join(translated_parts)
                translation_sec        = round(time.perf_counter() - t0, 3)
                log["translated_text"]  = final_text
                log["translation_sec"]  = translation_sec
                yield {"event": "translation_done", "data": _sse_json({
                    "final_text":      final_text,
                    "translation_sec": translation_sec,
                })}
            else:
                final_text = refined_text
                log["translation_sec"] = 0.0  # not needed — already English, nothing to translate

            # `final_text` is the exact text handed to the extraction step below —
            # this is what fills the template's "Patient Story" field, verbatim,
            # with no further LLM processing.
            patient_story = final_text
            routing_signal = detect_red_flags("\n".join((text, refined_text, final_text)))
            log["routing_signal"] = routing_signal
            yield {"event": "routing_signal", "data": _sse_json(routing_signal)}

            # ── 3. Field extraction ‖ ICD-10 diagnosis — run in parallel ─────────
            async def run_extraction():
                meta: dict = {}
                t0 = time.perf_counter()
                extracted = await LLMService.extract_features_async(
                    translated_text=final_text,
                    runtime=runtime,
                    department=department_value,
                    is_conversation=is_conversation,
                    model_id=extraction_spec.id,
                    meta=meta,
                )
                raw_features = extracted.get("json_data", {}) if extracted else {}
                # ── Phase 2: coerce LLM output against the department extraction schema ──
                # Never-fail: returns raw_features unchanged if validation errors occur.
                features = validate_extraction_output(raw_features, department_value)
                t_q = time.perf_counter()
                # Questions derive deterministically from the extracted fields (see QuestionGenerator).
                questions, _ = QuestionGenerator.generate(extracted_features=features, department=department_value)
                return {
                    "features": features, "questions": questions, "meta": meta,
                    "extraction_sec": round(t_q - t0, 3),
                    "question_generation_sec": round(time.perf_counter() - t_q, 3),
                }

            async def run_diagnosis():
                meta: dict = {}
                t0 = time.perf_counter()
                catalogue_candidates = get_registry().search_diagnosis_candidates(
                    final_text
                )
                diagnosis_result = await LLMService.recommend_diagnosis_async(
                    translated_text=final_text,
                    runtime=runtime,
                    department=department_value,
                    patient_history_text=patient_history_text,
                    is_conversation=is_conversation,
                    model_id=reasoning_spec.id,
                    meta=meta,
                    catalogue_candidates=catalogue_candidates,
                )
                raw_data = diagnosis_result.get("json_data", {}) if diagnosis_result else {}
                # ── Phase 2: coerce LLM diagnosis output against DiagnosisOutput schema ──
                # Never-fail: returns raw_data unchanged if validation errors occur.
                data = validate_diagnosis_output(raw_data)
                data = constrain_diagnosis_to_registry(
                    data,
                    department_value,
                    catalogue_candidates,
                )
                return {"data": data, "meta": meta, "diagnosis_sec": round(time.perf_counter() - t0, 3)}

            errors: dict = {}
            extracted_features, raw_questions, diagnosis_data = {}, [], {}
            extraction_sec = question_generation_sec = diagnosis_sec = None
            ext_meta = diag_meta = None
            history_check = None

            async def capture_step(step: str, operation):
                try:
                    return step, await operation(), None
                except Exception as exc:
                    return step, None, exc

            parallel_tasks = [
                asyncio.create_task(capture_step("extraction", run_extraction)),
                asyncio.create_task(capture_step("diagnosis", run_diagnosis)),
            ]

            # Send each step as soon as it finishes; the slower parallel step
            # continues running while the browser renders the completed one.
            for completed in asyncio.as_completed(parallel_tasks):
                step, result, error = await completed
                if step == "extraction":
                    if error:
                        errors[step] = str(error)
                        logger.error("[Analyze] extraction failed: %s", error)
                        event_data = {
                            "department": department_value,
                            "model": extraction_spec.label,
                            "error": str(error),
                        }
                    else:
                        extracted_features = result["features"]
                        raw_questions = result["questions"]
                        extraction_sec = result["extraction_sec"]
                        question_generation_sec = result["question_generation_sec"]
                        ext_meta = result["meta"]
                        event_data = {
                            "department": department_value,
                            "patient_story": patient_story,
                            "routing_signal": routing_signal,
                            "extracted_features": extracted_features,
                            "extraction_sec": extraction_sec,
                            "questions": raw_questions,
                            "question_generation_sec": question_generation_sec,
                            "model": extraction_spec.label,
                            "usage": ext_meta,
                        }
                    yield {"event": "extraction", "data": json.dumps(event_data)}
                else:
                    if error:
                        errors[step] = str(error)
                        logger.error("[Analyze] diagnosis failed: %s", error)
                        event_data = {
                            "department": department_value,
                            "routing_signal": routing_signal,
                            "model": reasoning_spec.label,
                            "error": str(error),
                            "history_used": bool(patient_history_text),
                            "history_visit_count": len(patient_history_visits),
                            "history_error": history_error,
                        }
                    else:
                        diagnosis_data = result["data"]
                        diagnosis_sec = result["diagnosis_sec"]
                        diag_meta = result["meta"]
                        if patient_history_text and diagnosis_data:
                            history_check = check_history_usage(patient_history_visits, diagnosis_data)
                            if history_check["flag"] == "claims_used_no_overlap":
                                logger.warning(
                                    "[Analyze] patient_code=%s: history_considered=Yes but no ICD-10 family overlap — history_notes=%r",
                                    patient_id, diagnosis_data.get("history_notes"),
                                )
                            elif history_check["flag"] == "overlap_but_not_flagged":
                                logger.warning(
                                    "[Analyze] patient_code=%s: ICD-10 family overlap %s but history_considered was not 'Yes'",
                                    patient_id, history_check["overlap_families"],
                                )
                        event_data = {
                            "department": department_value,
                            "diagnosis": diagnosis_data,
                            "diagnosis_sec": diagnosis_sec,
                            "model": reasoning_spec.label,
                            "usage": diag_meta,
                            "history_used": bool(patient_history_text),
                            "history_visit_count": len(patient_history_visits),
                            "history_check": history_check,
                            "history_error": history_error,
                        }
                    yield {"event": "diagnosis", "data": json.dumps(event_data)}

            log["extracted_features"]       = extracted_features
            log["questions"]                = raw_questions
            log["diagnosis"]                = diagnosis_data
            log["extraction_sec"]           = extraction_sec
            log["question_generation_sec"]  = question_generation_sec
            log["diagnosis_sec"]            = diagnosis_sec
            log["extraction_usage"]         = ext_meta
            log["diagnosis_usage"]          = diag_meta
            log["history_check"]            = history_check
            if errors:
                log["status"] = "error"

        finally:
            for task in parallel_tasks:
                if not task.done():
                    task.cancel()
            if parallel_tasks:
                await asyncio.gather(*parallel_tasks, return_exceptions=True)
            log["total_sec"] = round(time.perf_counter() - pipeline_start, 3)
            session_repo: Optional[SessionRepository] = getattr(request.app.state, "sessions", None)
            await persist_session(log, session_repo)

    return EventSourceResponse(
        event_stream(),
        headers={
            # Disable Nginx's response buffering for this stream specifically —
            # without this, a reverse proxy sitting in front of the app can
            # hold the whole SSE response in its buffer and flush it late (or
            # drop/merge frames under load) instead of forwarding each event
            # as it's yielded, which is what was causing the 'extraction'
            # event — by far the largest one in this stream — to sometimes
            # never reach the browser at all.
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
        },
    )


app.include_router(router)
app.include_router(patient_router)
app.include_router(eval_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7777)
