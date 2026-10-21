import os
import time
import json
import logging
import asyncio
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

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
from pipeline import PipelineContext, run_pipeline, MergedResult

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
from core.auth import (
    get_current_user,
    require_admin,
    require_dashboard_access,
    require_authenticated,
    AuthenticatedUser,
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


def sanitize_usage_entry(entry: dict) -> dict:
    """
    PHI-Safe Log Sanitizer (Milestone 7 / M7.2):
    Removes raw transcripts, refined text, and unmasked clinical free-text from
    JSONL logs. Preserves operational telemetry, latencies, tokens, department mix,
    and a one-way SHA-256 integrity hash for auditing without leaking patient PHI.
    """
    clean = dict(entry)
    raw_text = clean.pop("transcript", None)
    clean.pop("refined_text", None)
    clean.pop("translated_text", None)
    clean.pop("patient_story", None)

    # Compute deterministic, irreversible integrity hash if raw text was present
    if raw_text:
        import hashlib
        clean["transcript_hash"] = hashlib.sha256(str(raw_text).encode("utf-8")).hexdigest()
    else:
        clean["transcript_hash"] = None

    clean["phi_safe"] = True
    return clean


def save_usage_log(entry: dict) -> None:
    try:
        sanitized = sanitize_usage_entry(entry)
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
        logger.info("[UsageLog] Saved PHI-safe entry for %s", sanitized.get("ip"))
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


def _require_dashboard_auth(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(_dashboard_security),
) -> AuthenticatedUser:
    """
    Milestone 7 (M7.1) Integrated Dashboard Auth:
    - If APIM_ENABLED=true: Enforces Entra ID identity from APIM headers.
      Allows ClinicalAdmin (full ops) and ExecutiveStakeholder (financial/analytics tracking).
      Rejects standard Doctor or unauthenticated users with 401/403.
    - If APIM_ENABLED=false: Supports dev/staging mode. If DASHBOARD_USERNAME/PASSWORD
      are configured and credentials provided, validates them; otherwise checks emulated headers
      or query params (?role=ClinicalAdmin or ?role=ExecutiveStakeholder).
    """
    apim_enabled = os.getenv("APIM_ENABLED", "false").lower() == "true"
    if apim_enabled:
        user = get_current_user(request)
        if not user.can_view_dashboard:
            raise HTTPException(
                status_code=403,
                detail="Access denied: ClinicalAdmin or ExecutiveStakeholder role required",
            )
        return user

    # Dev / Staging Mode:
    dash_user = getattr(Config, "DASHBOARD_USERNAME", None)
    dash_pass = getattr(Config, "DASHBOARD_PASSWORD", None)

    # 1. Check if user passed basic auth credentials in dev
    if credentials and dash_user and dash_pass:
        ok = secrets.compare_digest(credentials.username, dash_user) and secrets.compare_digest(credentials.password, dash_pass)
        if ok:
            return AuthenticatedUser(
                user_id="dev-basic-auth",
                email=f"{dash_user}@andalusiagroup.net",
                display_name=dash_user,
                roles=("ClinicalAdmin", "ExecutiveStakeholder"),
            )
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    # 2. Check role emulation in dev (headers or ?role=)
    dev_roles = request.headers.get("x-user-roles") or request.query_params.get("role")
    if dev_roles:
        roles = tuple(r.strip() for r in dev_roles.split(",") if r.strip())
        user = AuthenticatedUser(
            user_id=request.headers.get("x-user-id", "dev-emulator-id"),
            email=request.headers.get("x-user-email", "dev@localhost"),
            display_name=request.headers.get("x-user-name", "Dev User"),
            roles=roles,
        )
        if not user.can_view_dashboard:
            raise HTTPException(
                status_code=403,
                detail="Access denied: ClinicalAdmin or ExecutiveStakeholder role required",
            )
        return user

    # If neither basic credentials nor role headers were supplied:
    if dash_user and dash_pass:
        raise HTTPException(status_code=401, detail="Authentication required", headers={"WWW-Authenticate": "Basic"})

    user = get_current_user(request)
    if not user.can_view_dashboard:
        raise HTTPException(
            status_code=403,
            detail="Access denied: ClinicalAdmin or ExecutiveStakeholder role required",
        )
    return user


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


@app.get("/admin/me", summary="Current user profile and dashboard role capabilities")
async def admin_me(user: AuthenticatedUser = Depends(_require_dashboard_auth)):
    return {
        "user_id": user.user_id,
        "email": user.email,
        "display_name": user.display_name,
        "roles": user.roles,
        "is_admin": user.is_admin,
        "is_stakeholder": user.is_stakeholder,
        "can_view_dashboard": user.can_view_dashboard,
    }


@app.get("/admin/financial-summary", summary="Aggregated financial metrics, AI costs, and ROI calculation")
async def admin_financial_summary(
    request: Request,
    days: int = 30,
    department: Optional[str] = None,
    auth_user: AuthenticatedUser = Depends(_require_dashboard_auth),
):
    """
    Task 2.2: High-Performance Financial & Observability Aggregation Service
    Returns pre-aggregated metrics: total AI infrastructure cost, hours saved,
    net financial savings, net ROI %, department breakdown, and daily burn rate.
    """
    from datetime import timedelta
    rates = {
        "input_cost_per_m": getattr(Config, "LLM_INPUT_COST_PER_M", 0.27),
        "output_cost_per_m": getattr(Config, "LLM_OUTPUT_COST_PER_M", 1.10),
        "stt_cost_per_min": getattr(Config, "STT_COST_PER_MINUTE", 0.006),
        "clinician_hourly_rate": getattr(Config, "CLINICIAN_HOURLY_RATE_USD", 75.0),
        "manual_minutes": getattr(Config, "MANUAL_DOCUMENTATION_MINUTES", 8.5),
        "ai_minutes": getattr(Config, "AI_DOCUMENTATION_MINUTES", 2.0),
    }

    session_repo: Optional[SessionRepository] = getattr(request.app.state, "sessions", None)
    if session_repo is not None:
        try:
            return await session_repo.get_financial_summary(days=days, department=department, rates=rates)
        except Exception as exc:
            logger.warning("[Dashboard] Mongo aggregation pipeline failed, falling back to JSONL calculation: %s", exc)

    # High-performance in-memory stream fallback from JSONL
    entries = _read_jsonl(USAGE_LOG, limit=5000)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days) if days > 0 else None

    tot_sessions = 0
    success_sessions = 0
    p_tokens = 0
    c_tokens = 0
    audio_sec = 0.0
    dept_map: Dict[str, Dict[str, Any]] = {}
    daily_map: Dict[str, Dict[str, Any]] = {}

    for e in entries:
        ts = e.get("timestamp")
        if ts and cutoff:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt < cutoff:
                    continue
            except Exception:
                pass
        dep = e.get("department") or "unknown"
        if department and dep != department:
            continue

        tot_sessions += 1
        if e.get("status") == "ok":
            success_sessions += 1

        ext_u = e.get("extraction_usage") or {}
        diag_u = e.get("diagnosis_usage") or {}
        p_tok = (ext_u.get("prompt_tokens") or 0) + (diag_u.get("prompt_tokens") or 0)
        c_tok = (ext_u.get("completion_tokens") or 0) + (diag_u.get("completion_tokens") or 0)
        a_sec = float(e.get("transcription_sec") or 0.0)

        p_tokens += p_tok
        c_tokens += c_tok
        audio_sec += a_sec

        cost = ((p_tok / 1_000_000.0) * rates["input_cost_per_m"]
                + (c_tok / 1_000_000.0) * rates["output_cost_per_m"]
                + (a_sec / 60.0) * rates["stt_cost_per_min"])

        if dep not in dept_map:
            dept_map[dep] = {"sessions": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        dept_map[dep]["sessions"] += 1
        dept_map[dep]["prompt_tokens"] += p_tok
        dept_map[dep]["completion_tokens"] += c_tok
        dept_map[dep]["cost_usd"] = round(dept_map[dep]["cost_usd"] + cost, 4)

        date_key = ts[:10] if ts and len(ts) >= 10 else "unknown"
        if date_key not in daily_map:
            daily_map[date_key] = {"sessions": 0, "cost_usd": 0.0}
        daily_map[date_key]["sessions"] += 1
        daily_map[date_key]["cost_usd"] = round(daily_map[date_key]["cost_usd"] + cost, 4)

    llm_cost = round((p_tokens / 1_000_000.0) * rates["input_cost_per_m"] + (c_tokens / 1_000_000.0) * rates["output_cost_per_m"], 4)
    stt_cost = round((audio_sec / 60.0) * rates["stt_cost_per_min"], 4)
    total_ai_cost = round(llm_cost + stt_cost, 4)

    mins_saved = max(0.0, rates["manual_minutes"] - rates["ai_minutes"])
    hours_saved = round((success_sessions * mins_saved) / 60.0, 2)
    gross_value = round(hours_saved * rates["clinician_hourly_rate"], 2)
    net_savings = round(gross_value - total_ai_cost, 2)
    roi_percent = round((net_savings / total_ai_cost * 100.0) if total_ai_cost > 0 else 0.0, 1)

    return {
        "period_days": days,
        "total_sessions": tot_sessions,
        "successful_sessions": success_sessions,
        "token_usage": {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        },
        "audio_duration_minutes": round(audio_sec / 60.0, 2),
        "costs": {
            "llm_cost_usd": llm_cost,
            "stt_cost_usd": stt_cost,
            "total_ai_cost_usd": total_ai_cost,
            "cost_per_session_usd": round(total_ai_cost / tot_sessions, 4) if tot_sessions > 0 else 0.0,
        },
        "savings": {
            "hours_saved": hours_saved,
            "gross_value_saved_usd": gross_value,
            "net_savings_usd": net_savings,
            "net_roi_percent": roi_percent,
        },
        "by_department": dept_map,
        "daily_burn": dict(sorted(daily_map.items())),
        "source": "jsonl_stream",
    }


@app.get("/admin/usage", summary="Recent /audio/analyze usage log entries, newest first")
async def admin_usage(
    request: Request,
    limit: int = 500,
    department: Optional[str] = None,
    auth_user: AuthenticatedUser = Depends(_require_dashboard_auth),
):
    limit = max(1, min(limit, 5000))
    session_repo: Optional[SessionRepository] = getattr(request.app.state, "sessions", None)
    entries: List[Dict[str, Any]] = []
    if session_repo is not None:
        try:
            entries = await session_repo.list_sessions(limit=limit, department=department)
            source = "mongodb"
        except Exception as exc:
            logger.warning("[Dashboard] Failed to read sessions from MongoDB, falling back to JSONL: %s", exc)
            entries = _read_jsonl(USAGE_LOG, limit)
            source = "jsonl"
    else:
        entries = _read_jsonl(USAGE_LOG, limit)
        source = "jsonl"

    if department:
        entries = [e for e in entries if e.get("department") == department]

    # For ExecutiveStakeholder (non-admin), mask individual clinician identifiers (PHI/PII safe)
    if not auth_user.is_admin:
        sanitized = []
        for e in entries:
            sc = dict(e)
            sc.pop("clinician_name", None)
            sc.pop("sso_user", None)
            sc.pop("doctor_id", None)
            sc["clinician_id"] = "ANONYMIZED_CLINICIAN"
            sanitized.append(sc)
        entries = sanitized

    return {"entries": entries, "source": source}


@app.get("/admin/feedback", summary="Recent doctor feedback entries, newest first")
async def admin_feedback(
    request: Request,
    limit: int = 500,
    department: Optional[str] = None,
    auth_user: AuthenticatedUser = Depends(_require_dashboard_auth),
):
    limit = max(1, min(limit, 5000))
    feedback_repo: Optional[FeedbackRepository] = getattr(request.app.state, "feedback_repo", None)
    entries: List[Dict[str, Any]] = []
    if feedback_repo is not None:
        try:
            entries = await feedback_repo.list_recent(limit=limit, department=department)
            source = "mongodb"
        except Exception as exc:
            logger.warning("[Dashboard] Failed to read feedback from MongoDB, falling back to JSONL: %s", exc)
            entries = _read_jsonl(FEEDBACK_LOG, limit)
            source = "jsonl"
    else:
        entries = _read_jsonl(FEEDBACK_LOG, limit)
        source = "jsonl"

    if department:
        entries = [e for e in entries if e.get("department") == department]

    if not auth_user.is_admin:
        sanitized = []
        for e in entries:
            sc = dict(e)
            sc.pop("sso_user", None)
            sc.pop("doctor_id", None)
            sc["name"] = "ANONYMIZED_CLINICIAN"
            sanitized.append(sc)
        entries = sanitized

    return {"entries": entries, "source": source}


@app.get("/dashboard", summary="Analytics dashboard — usage + feedback, for internal review")
async def serve_dashboard(auth_user: AuthenticatedUser = Depends(_require_dashboard_auth)):
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
    started = time.perf_counter()
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


# ── Dependency health (A7.1.1) ─────────────────────────────────────────────
# Cached for 30 s so a burst of smoke-test/availability probes does not hammer
# the providers. The cache stores only status/latency — never secrets.
_HEALTH_CACHE_TTL_S = 30.0
_health_cache: Dict[str, Any] = {"at": 0.0, "payload": None}


async def _probe_foundry(request: Request) -> dict:
    """Probe Foundry with a lightweight authenticated completion (no patient data).

    Keep the health-specific probe here rather than changing the provider
    adapter's existing health_check contract. ``models.list()`` can take longer
    than the health budget; a tiny chat completion uses the shared runtime,
    standard timeout/retry policy and the configured extraction deployment.
    """
    from providers.factory import get_llm
    provider = getattr(request.app.state, "llm_provider", None) or get_llm()
    started = time.perf_counter()
    try:
        runtime = getattr(provider, "runtime", None)
        if runtime is None:
            return {"component": "foundry", "status": "error", "detail": "runtime_unavailable"}
        from llm.provider import build_foundry_kwargs
        deployment = Config.foundry_model_extraction()
        kwargs = build_foundry_kwargs(
            deployment,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16,
        )
        await runtime.complete(model=deployment, create_kwargs=kwargs)
        return {
            "component": "foundry",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    except Exception as exc:
        return {
            "component": "foundry",
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "detail": type(exc).__name__,
        }


async def _probe_stt(request: Request, provider_name: str) -> dict:
    """Probe an STT provider with a lightweight authenticated call (no audio).

    A failing STT provider is reported as "degraded" (not "error") so it never
    gates a deploy: only Foundry is a hard dependency for /health (owner decision).
    """
    from providers.factory import get_stt
    provider = getattr(request.app.state, "stt_provider", None) or get_stt()
    # Only probe the provider that is actually configured; report the other as
    # not_configured so a single-provider deployment is not failed by the other.
    active = getattr(provider, "provider_name", os.getenv("STT_PROVIDER", "munsit"))
    if active != provider_name:
        return {"component": provider_name, "status": "not_configured"}
    try:
        result = await provider.health_check()
        return {
            "component": provider_name,
            "status": "ok" if result.ok else "degraded",
            "latency_ms": result.latency_ms,
        }
    except Exception as exc:
        return {"component": provider_name, "status": "degraded", "detail": type(exc).__name__}


async def _dependency_health(request: Request) -> dict:
    """Return cached per-dependency health for Foundry, Munsit and Voxtral.

    HTTP gating (owner decision): only Foundry is a hard dependency. Foundry
    unreachable or its key/token rejected -> ``unhealthy`` (HTTP 503). A Munsit
    or Voxtral problem is reported as ``degraded`` but keeps HTTP 200, so an STT
    provider outage does not fail a deploy.
    """
    now = time.monotonic()
    cached = _health_cache.get("payload")
    if cached is not None and (now - _health_cache.get("at", 0.0)) < _HEALTH_CACHE_TTL_S:
        return cached

    foundry, munsit, voxtral = await asyncio.gather(
        _probe_foundry(request),
        _probe_stt(request, "munsit"),
        _probe_stt(request, "voxtral"),
    )
    checks = [foundry, munsit, voxtral]
    if foundry["status"] not in ("ok",):
        overall = "unhealthy"
    elif any(c["status"] == "degraded" for c in checks):
        overall = "degraded"
    else:
        overall = "ok"
    payload = {
        "status": overall,
        "checks": checks,
    }
    _health_cache["at"] = now
    _health_cache["payload"] = payload
    return payload


@app.get("/health", tags=["Observability"])
async def health_check(request: Request):
    """Dependency health probe for deploy smoke tests (A7.1.1).

    Returns 200 with per-dependency status for Foundry, Munsit and Voxtral, or
    503 when Foundry is unreachable or its key/token is rejected. Results are
    cached for 30 s. No secrets or endpoints appear in the response.

    A Munsit or Voxtral problem is reported as ``degraded`` but keeps HTTP 200,
    so an STT provider outage does not fail a deploy. /health/live remains the
    dependency-free liveness probe.
    """
    result = await _dependency_health(request)
    status_code = 200 if result["status"] in ("ok", "degraded") else 503
    return Response(
        content=json.dumps(result, separators=(",", ":")),
        status_code=status_code,
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
        extraction_spec = get_model("extraction", inp.extraction_model)
        reasoning_spec  = get_model("reasoning",  inp.reasoning_model)
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

    client_ip = _get_client_ip(request)
    client_user = _get_client_user(request)
    user_agent = request.headers.get("user-agent", "unknown")
    app_obj = getattr(request, "app", None) if "app" in request.scope else None
    session_repo: Optional[SessionRepository] = getattr(app_obj.state, "sessions", None) if app_obj else None

    ctx = PipelineContext(
        session_id=session_id,
        text=text,
        language=language,
        mode=mode,
        department_value=department_value,
        specialty_value=specialty_value,
        runtime=runtime,
        extraction_spec=extraction_spec,
        reasoning_spec=reasoning_spec,
        patient_id=patient_id,
        business_unit_value=inp.business_unit.value if inp.business_unit else None,
        use_history=use_history,
        patient_history_visits=patient_history_visits,
        patient_history_text=patient_history_text,
        history_error=history_error,
        transcription_sec=transcription_sec,
        clinician_name=clinician_name,
        pipeline_start=pipeline_start,
    )

    logger.info(
        "[Analyze] session start — clinician_name=%r sso_user=%r department=%s language=%s mode=%s",
        clinician_name, client_user, department_value, language.value, mode.value,
    )

    async def _on_pipeline_complete(merged: MergedResult) -> None:
        log = merged.to_log_dict(
            ip=client_ip,
            user_agent=user_agent,
            clinician_name=clinician_name,
            sso_user=client_user,
        )
        await persist_session(log, session_repo)

    return EventSourceResponse(
        run_pipeline(ctx, on_complete=_on_pipeline_complete),
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
