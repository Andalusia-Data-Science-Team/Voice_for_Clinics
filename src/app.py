# ── Phase 0: Key Vault bootstrap — MUST be before any other import ──────────
from core.azure_identity import bootstrap as _kv_bootstrap; _kv_bootstrap()
# ─────────────────────────────────────────────────────────────────────────────
import time
import json
import logging
import asyncio
import sys
import secrets

# On Windows, asyncio subprocesses (like ffmpeg piping) require ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
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
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

# Phase 1: APIM-native auth — no JWT library, no BasicAuth
from core.auth import (
    AuthenticatedUser,
    get_current_user,
    get_sso_user,
    require_admin,
    require_authenticated,
)

# Phase 2: Application Insights telemetry
from core.telemetry import (
    init_telemetry,
    track_event,
    increment_active_analyses,
    decrement_active_analyses,
    increment_active_websockets,
    decrement_active_websockets,
)

# Phase 4: Admission controller — capacity gating + health probe
from core.admission import AdmissionController, AdmissionRejectedError, admission_controller

from core.config import Config
from model.speech_service import SpeechService
from model.llm_service import LLMService
from model.question_generator import QuestionGenerator
from core.model_registry import get_model, registry_for_ui
from model.specialty_terms import correct_specialty_terms, SPECIALTIES
from model.patient_history import (
    get_patient_visits,
    format_history_for_prompt,
    check_history_usage,
    BusinessUnit,
    PatientHistoryError,
)
from llm.runtime import LLMRuntime, get_llm_runtime


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio Processing"])
patient_router = APIRouter(prefix="/patient", tags=["Patient History"])


def get_current_request(request: Request) -> Request:
    """Provide the current Starlette request through FastAPI dependency injection."""
    return request


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and close one shared asynchronous LLM runtime per worker."""
    # Secrets already in os.environ (bootstrap() ran at module load).
    # Initialize telemetry (Phase 2)
    init_telemetry()
    # LLMRuntime.__init__ reads Config.llm_api_key() lazily — safe here.
    app.state.llm = LLMRuntime()
    try:
        yield
    finally:
        await app.state.llm.close()


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

# ── Phase 1: CORS is handled by APIM global.xml ──────────────────────────────
# CORSMiddleware has been removed from FastAPI.
# APIM's <cors> policy in apim/policies/global.xml handles all preflight and
# CORS headers.  Keeping it here would double-set headers and confuse browsers.
#
# Dev mode without APIM: run with APIM_ENABLED=false — browsers that hit
# FastAPI directly (e.g. via `uvicorn app:app`) will need a local proxy or
# browser CORS-disabled mode, OR re-enable CORSMiddleware temporarily by
# setting DEV_CORS=true (checked below).
import os as _os
if _os.getenv("DEV_CORS", "false").lower() == "true":
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.warning("[CORS] DEV_CORS=true — permissive CORS active (dev only, never production)")


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
    Phase 1: Returns the SSO user from APIM-injected X-User-* headers.
    In dev mode (APIM_ENABLED=false) falls back to legacy proxy headers
    (X-Forwarded-User etc.) for smooth staging transition.
    See core.auth.get_sso_user() for the full resolution logic.
    """
    return get_sso_user(request)


def save_usage_log(entry: dict) -> None:
    try:
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("[UsageLog] Saved entry for %s", entry.get("ip"))
    except Exception as e:
        logger.warning("[UsageLog] Failed to write log: %s", e)

    # Phase 2: Send structured AnalyzeSession event to Application Insights.
    # PHI text fields are excluded (enforced both here and inside track_event).
    try:
        properties = {
            "session_id": entry.get("session_id"),
            "ip": entry.get("ip"),
            "sso_user": entry.get("sso_user"),
            "clinician_name": entry.get("clinician_name"),
            "language": entry.get("language"),
            "mode": entry.get("mode"),
            "department": entry.get("department"),
            "specialty": entry.get("specialty"),
            "patient_id": entry.get("patient_id"),
            "business_unit": entry.get("business_unit"),
            "use_history": str(entry.get("use_history")),
            "status": entry.get("status"),
            "refinement_skipped": str(entry.get("refinement_skipped")),
        }
        measurements = {
            "total_sec": float(entry["total_sec"]) if entry.get("total_sec") is not None else 0.0,
            "transcription_sec": float(entry["transcription_sec"]) if entry.get("transcription_sec") is not None else 0.0,
            "refinement_sec": float(entry["refinement_sec"]) if entry.get("refinement_sec") is not None else 0.0,
            "translation_sec": float(entry["translation_sec"]) if entry.get("translation_sec") is not None else 0.0,
            "extraction_sec": float(entry["extraction_sec"]) if entry.get("extraction_sec") is not None else 0.0,
            "question_generation_sec": float(entry["question_generation_sec"]) if entry.get("question_generation_sec") is not None else 0.0,
            "history_visit_count": float(entry.get("history_visit_count") or 0),
        }
        track_event("AnalyzeSession", properties=properties, measurements=measurements)
    except Exception:
        pass


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


@app.get("/models", summary="Fixed models used by the analysis pipeline")
async def list_models():
    return registry_for_ui()


@app.post("/feedback", summary="Submit doctor feedback (thumbs rating and/or free text)")
async def submit_feedback(
    request:    Request,
    rating:     Optional[str] = Form(None, description='"up", "down", or omitted for no rating'),
    text:       str           = Form("",   description="Free-text feedback, complaint, or enhancement suggestion"),
    name:       Optional[str] = Form(None, description="Self-reported name of the person giving feedback"),
    department: Optional[str] = Form(None),
    language:   Optional[str] = Form(None),
    session_id: Optional[str] = Form(None, description="The record→analyze session this feedback is about, if any — joins back to that usage-log entry on the dashboard"),
):
    text = (text or "").strip()
    if rating not in (None, "", "up", "down"):
        raise HTTPException(status_code=400, detail='rating must be "up" or "down"')
    rating = rating or None
    if not text and not rating:
        raise HTTPException(status_code=400, detail="Provide feedback text and/or a thumbs rating.")

    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "ip":         _get_client_ip(request),
        "name":       (name or "").strip() or None,
        "sso_user":   _get_client_user(request),
        "rating":     rating,
        "text":       text,
        "department": department or None,
        "language":   language or None,
    }
    logger.info(
        "[Feedback] session_id=%s name=%s rating=%s department=%s language=%s text=%r",
        entry["session_id"], entry["name"], entry["rating"], entry["department"], entry["language"], entry["text"],
    )
    await asyncio.to_thread(save_feedback_log, entry)

    # Phase 2: Send structured DoctorFeedback event to Application Insights
    try:
        track_event(
            "DoctorFeedback",
            properties={
                "session_id": entry["session_id"],
                "name": entry["name"],
                "sso_user": entry["sso_user"],
                "rating": entry["rating"],
                "department": entry["department"],
                "language": entry["language"],
                "has_text": str(bool(entry["text"])),
            },
        )
    except Exception:
        pass

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
# ⚠ ACCESS CONTROL (Phase 1):
#   In production: APIM admin.xml enforces ClinicalAdmin role at the gateway
#   BEFORE the request reaches FastAPI.  require_admin() below is the
#   FastAPI-side defence-in-depth duplicate check.
#   In dev (APIM_ENABLED=false): ANONYMOUS has ClinicalAdmin, so all routes
#   are accessible locally without any credentials.
# ═══════════════════════════════════════════════════════════════════════════════


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
        logger.warning("[Dashboard] Failed to read %s: %s", path, e)
        return []
    entries.reverse()
    return entries[:limit]


@app.get("/admin/usage", summary="Recent /audio/analyze usage log entries, newest first")
async def admin_usage(
    limit: int = 500,
    user: AuthenticatedUser = Depends(require_admin),
):
    """ClinicalAdmin only. Returns up to `limit` usage log entries, newest first."""
    limit = max(1, min(limit, 5000))
    logger.info("[Dashboard] admin_usage accessed by %s", user.display)
    return {"entries": _read_jsonl(USAGE_LOG, limit)}


@app.get("/admin/feedback", summary="Recent doctor feedback entries, newest first")
async def admin_feedback(
    limit: int = 500,
    user: AuthenticatedUser = Depends(require_admin),
):
    """ClinicalAdmin only. Returns up to `limit` feedback log entries, newest first."""
    limit = max(1, min(limit, 5000))
    logger.info("[Dashboard] admin_feedback accessed by %s", user.display)
    return {"entries": _read_jsonl(FEEDBACK_LOG, limit)}


@app.get("/dashboard", summary="Analytics dashboard — usage + feedback, for internal review")
async def serve_dashboard(user: AuthenticatedUser = Depends(require_admin)):
    """ClinicalAdmin only. Serves the static dashboard HTML."""
    logger.info("[Dashboard] dashboard accessed by %s", user.display)
    return FileResponse(Path(__file__).parent / "static" / "dashboard.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    arabic  = "ar"
    english = "en"


class Mode(str, Enum):
    doctor       = "doctor"
    conversation = "conversation"


class Department(str, Enum):
    """
    Selects which clinical note template/schema the extraction step targets.
    Each maps 1:1 to a Word template: ER_Clinical_Note_Template.docx,
    OPD_Clinical_Note_Template.docx, Dental_Clinical_Note_Template.docx.
    """
    er        = "er"
    opd       = "opd"
    dental    = "dental"


# Built from model.specialty_terms.SPECIALTIES so the two never drift apart.
Specialty = Enum("Specialty", {s.replace(" ", "_").replace("&", "and"): s for s in SPECIALTIES}, type=str)


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
        audio_bytes = await SpeechService.get_recording_disclaimer_audio(Config.munsit_api_key())
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
    # Phase 4: Admission gate — check capacity BEFORE accepting the socket.
    # If we are saturated we reject at the TCP level (close before accept),
    # which is the cleanest signal a client can receive.
    try:
        await admission_controller.check_websocket()
    except AdmissionRejectedError as exc:
        # Accept then immediately close with a 1008 (Policy Violation) code
        # so the browser-side WebSocket gets an error event rather than hanging.
        await websocket.accept()
        await websocket.close(code=1008, reason=str(exc))
        logger.warning("[WS] Admission rejected: %s", exc)
        return

    await websocket.accept()
    increment_active_websockets()
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
                api_key=Config.munsit_api_key(),
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
                api_key=Config.mistral_api_key(),
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
        decrement_active_websockets()
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
# GET /audio/icd10/search — sub-10ms autocomplete search across 16,954 Australian
# ICD-10-AM codes, supporting both prefix code matches ("J18") and fuzzy description
# queries ("pneumonia").
# ═══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/icd10/search",
    summary="Sub-10ms autocomplete search across Australian ICD-10-AM codes",
)
async def search_icd10(
    q: str = Query(..., min_length=1, description="Code prefix or description keyword"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results to return"),
    user: AuthenticatedUser = Depends(require_authenticated),
):
    """Sub-10ms fuzzy search across 16,954 Australian ICD-10-AM codes."""
    from model.icd10_registry import get_registry
    registry = get_registry()
    results = registry.search(q, limit=limit)
    return [{"code": r[0], "description": r[1]} for r in results]


@router.post(
    "/refine-field",
    summary="Refine one short spoken answer for a single clinical-note field",
)
async def refine_field_answer(
    runtime: LLMRuntime = Depends(get_llm_runtime),
    text: str = Form(..., description="Raw transcript of the doctor's spoken answer for one field"),
    language: Language = Form(Language.english, description="Language the answer was spoken in"),
    specialty: Optional[str] = Form(None, description="Medical specialty"),
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    specialty_value = specialty.value if specialty else None
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
    request:    Request,
    runtime:    LLMRuntime = Depends(get_llm_runtime),
    text:       str        = Form(...,               description="Raw transcript from live transcription (may include user edits)"),
    language:   Language   = Form(Language.english,  description="Audio language"),
    mode:       Mode       = Form(Mode.doctor,        description="Doctor or Conversation mode"),
    department: Department = Form(Department.opd,     description="Clinical note template to fill: ER, OPD, or Dental"),
    specialty:  Optional[Specialty] = Form(None,      description="Medical specialty — drives fuzzy term correction and refinement prompt context"),
    use_history: bool      = Form(False,              description="If true, look up the patient's visit history and factor it into icd10_codes/final_diagnosis"),
    patient_id: Optional[str] = Form(None,            description="Patient Code (the 'Code' field on the Patient Info card) — required when use_history is true"),
    business_unit: Optional[BusinessUnit] = Form(None, description="AHJ, AMH or ANC — which business unit's database holds this patient's history. Required when use_history is true"),
    clinician_name: Optional[str] = Form(None,        description="Self-reported doctor name, from the Patient Info 'Dr' field — best-effort session attribution"),
    transcription_sec: Optional[float] = Form(None,   description="Client-measured duration of the transcription session (recording start to STT completion) — this endpoint doesn't do transcription itself, so it can only log what the browser measured"),
    session_id: Optional[str] = Form(None,            description="Client-generated ID for this record→analyze→feedback cycle, so a later feedback submission can be joined back to this exact session on the dashboard"),
):
    pipeline_start  = time.perf_counter()
    is_arabic       = language == Language.arabic
    is_conversation = mode == Mode.conversation
    skip_refinement = mode == Mode.doctor  # Doctor mode: transcript goes straight through, no refinement step
    specialty_value = specialty.value if specialty else None
    department_value = department.value

    try:
        extraction_spec = get_model("extraction", None)
        reasoning_spec  = get_model("reasoning",  None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    # Patient history lookup — only when the doctor opted in. Opting in without
    # both a patient code and a business unit is a client error (the frontend
    # blocks it too), since we can't know which database to query.
    #
    # A lookup that FAILS (DB unreachable, bad credentials, ...) does not abort
    # the analysis: the note is still generated, just without history context,
    # and history_error is passed to the frontend so the doctor is told history
    # was requested but NOT applied — rather than silently looking like the
    # patient simply had no history.
    patient_history_visits: list = []
    patient_history_text   = ""
    history_error: Optional[str] = None
    if use_history:
        if not patient_id or not business_unit:
            raise HTTPException(
                status_code=400,
                detail="use_history requires both patient_id (Patient Code) and business_unit",
            )
        try:
            patient_history_visits = await asyncio.to_thread(get_patient_visits, patient_id, business_unit)
            patient_history_text   = format_history_for_prompt(patient_history_visits)
            logger.info(
                f"[Analyze] use_history=true for patient_code={patient_id!r} ({business_unit.value}) — "
                f"found {len(patient_history_visits)} prior visit(s)"
            )
        except PatientHistoryError as exc:
            history_error = str(exc)
            logger.error(
                f"[Analyze] History lookup failed for patient_code={patient_id!r} ({business_unit.value}) — "
                f"continuing without history: {exc}"
            )

    # Deterministic fuzzy-matching pass — corrects near-miss specialty terms
    # in the raw transcript before anything else. Runs regardless of mode, so
    # Doctor-mode transcripts (which skip LLM refinement entirely) still get
    # it; Conversation-mode transcripts get it again implicitly via the LLM
    # refinement prompt's specialty context below.
    text = correct_specialty_terms(text, specialty_value)

    # Phase 4: Pre-flight admission check — runs synchronously before the SSE
    # stream is opened so we can return a clean 503 JSON response instead of
    # starting the stream and then bailing, which confuses some SSE clients.
    try:
        await admission_controller.check_analysis()
    except AdmissionRejectedError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "capacity_exceeded",
                "message": str(exc),
                "active": exc.active,
                "max": exc.maximum,
                "retry_after_sec": 5,
            },
        )

    async def event_stream():
        increment_active_analyses()
        # Accumulated outputs — populated as each step completes
        log: dict = {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "session_id":         session_id,
            "ip":                 _get_client_ip(request),
            "user_agent":         request.headers.get("user-agent", "unknown"),
            "clinician_name":     (clinician_name or "").strip() or None,
            "sso_user":           _get_client_user(request),
            "language":           language.value,
            "mode":               mode.value,
            "department":         department_value,
            "specialty":          specialty_value,
            "patient_id":         patient_id,
            "business_unit":      business_unit.value if business_unit else None,
            "use_history":        use_history,
            "history_visit_count": len(patient_history_visits),
            "history_error":      history_error,
            "status":             "ok",
            "total_sec":          None,
            "transcription_sec":  transcription_sec,  # known immediately — client-measured, not computed by this endpoint
            "refinement_sec":     None,
            "translation_sec":    None,
            "extraction_sec":     None,
            "question_generation_sec": None,
            "transcript":         text,
            "refined_text":       None,
            "refinement_skipped": skip_refinement,
            "translated_text":    None,
            "extracted_features": None,
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

        try:
            # ── 1. Refine ─────────────────────────────────────────────────
            t0 = time.perf_counter()

            if skip_refinement:
                # Doctor mode: skip the refinement LLM call entirely and pass the
                # (possibly user-edited) transcript straight through to translate/extract.
                refined_text = text
                log["refined_text"] = refined_text
                log["refinement_sec"] = 0.0  # skipped entirely in Doctor mode — not "instant", just not run
                yield {"event": "refinement_skipped", "data": json.dumps({
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
                        yield {"event": "refinement_delta", "data": json.dumps({"delta": delta})}
                except Exception as exc:
                    log["status"] = "error"
                    yield {"event": "error", "data": json.dumps({"step": "refinement", "detail": str(exc)})}
                    return

                refined_text      = "".join(refined_parts)
                refinement_sec    = round(time.perf_counter() - t0, 3)
                log["refined_text"]   = refined_text
                log["refinement_sec"] = refinement_sec
                yield {"event": "refinement_done", "data": json.dumps({
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
                        yield {"event": "translation_delta", "data": json.dumps({"delta": delta})}
                except Exception as exc:
                    log["status"] = "error"
                    yield {"event": "error", "data": json.dumps({"step": "translation", "detail": str(exc)})}
                    return

                final_text             = "".join(translated_parts)
                translation_sec        = round(time.perf_counter() - t0, 3)
                log["translated_text"]  = final_text
                log["translation_sec"]  = translation_sec
                yield {"event": "translation_done", "data": json.dumps({
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
                features = extracted.get("json_data", {}) if extracted else {}
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
                diagnosis_result = await LLMService.recommend_diagnosis_async(
                    translated_text=final_text,
                    runtime=runtime,
                    department=department_value,
                    patient_history_text=patient_history_text,
                    is_conversation=is_conversation,
                    model_id=reasoning_spec.id,
                    meta=meta,
                )
                data = diagnosis_result.get("json_data", {}) if diagnosis_result else {}
                return {"data": data, "meta": meta, "diagnosis_sec": round(time.perf_counter() - t0, 3)}

            ext_res, diag_res = await asyncio.gather(run_extraction(), run_diagnosis(), return_exceptions=True)

            errors: dict = {}
            extracted_features, raw_questions, diagnosis_data = {}, [], {}
            extraction_sec = question_generation_sec = diagnosis_sec = None
            ext_meta = diag_meta = None

            if isinstance(ext_res, BaseException):
                errors["extraction"] = str(ext_res)
                logger.error("[Analyze] extraction failed: %s", ext_res)
            else:
                extracted_features, raw_questions = ext_res["features"], ext_res["questions"]
                extraction_sec, question_generation_sec = ext_res["extraction_sec"], ext_res["question_generation_sec"]
                ext_meta = ext_res["meta"]

            if isinstance(diag_res, BaseException):
                errors["diagnosis"] = str(diag_res)
                logger.error("[Analyze] diagnosis failed: %s", diag_res)
            else:
                diagnosis_data, diagnosis_sec, diag_meta = diag_res["data"], diag_res["diagnosis_sec"], diag_res["meta"]

            log["extracted_features"]       = extracted_features
            log["questions"]                = raw_questions
            log["diagnosis"]                = diagnosis_data
            log["extraction_sec"]           = extraction_sec
            log["question_generation_sec"]  = question_generation_sec
            log["diagnosis_sec"]            = diagnosis_sec
            log["extraction_usage"]         = ext_meta
            log["diagnosis_usage"]          = diag_meta

            if errors:
                log["status"] = "error"
                if len(errors) == 2:     # nothing to show — surface both failures and stop
                    for step, detail in errors.items():
                        yield {"event": "error", "data": json.dumps({"step": step, "detail": detail})}
                    return

            # History audit — history only ever feeds the diagnosis step now.
            history_check = None
            if patient_history_text and diagnosis_data:
                history_check = check_history_usage(patient_history_visits, diagnosis_data)
                if history_check["flag"] == "claims_used_no_overlap":
                    logger.warning(f"[Analyze] patient_code={patient_id}: history_considered=Yes but no ICD-10 family overlap "
                                   f"— history_notes={diagnosis_data.get('history_notes')!r}")
                elif history_check["flag"] == "overlap_but_not_flagged":
                    logger.warning(f"[Analyze] patient_code={patient_id}: ICD-10 family overlap {history_check['overlap_families']} "
                                   f"but history_considered was not 'Yes'")
            log["history_check"] = history_check

            payload = {
                "department":              department_value,
                "patient_story":           patient_story,
                # step 1
                "extracted_features":      extracted_features,
                "extraction_sec":          extraction_sec,
                "questions":               raw_questions,
                "question_generation_sec": question_generation_sec,
                # step 2
                "diagnosis":               diagnosis_data,
                "diagnosis_sec":           diagnosis_sec,
                "history_used":            bool(patient_history_text),
                "history_visit_count":     len(patient_history_visits),
                "history_check":           history_check,
                "history_error":           history_error,
                # meta
                "models":  {"extraction": extraction_spec.label, "reasoning": reasoning_spec.label},
                "usage":   {"extraction": ext_meta, "diagnosis": diag_meta},
                "errors":  errors,
                "total_sec": round(time.perf_counter() - pipeline_start, 3),
            }
            logger.info("[Analyze] Emitting combined 'extraction' SSE event (extraction=%s, diagnosis=%s, %d question(s), ~%d bytes)",
                        "ok" if "extraction" not in errors else "FAILED",
                        "ok" if "diagnosis" not in errors else "FAILED",
                        len(raw_questions), len(json.dumps(payload)))
            yield {"event": "extraction", "data": json.dumps(payload)}

        finally:
            decrement_active_analyses()
            log["total_sec"] = round(time.perf_counter() - pipeline_start, 3)
            await asyncio.to_thread(save_usage_log, log)

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


# ═══════════════════════════════════════════════════════════════════════════════
# GET /health — APIM health probe + Azure Monitor liveness endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", summary="Liveness + capacity health probe for APIM and Azure Monitor")
async def health_probe(request: Request):
    """
    Returns HTTP 200 (status=ok) or HTTP 503 (status=degraded) with a JSON
    body that Azure API Management can use as a backend health check.

    Response body:
      status              : "ok" | "degraded"
      timestamp           : ISO-8601 UTC
      version             : application version label
      admission           : concurrency stats from AdmissionController
        active_analyses       / max_analyses    / analyses_utilization
        active_websockets     / max_websockets  / websockets_utilization
        is_saturated          : true when either limit is reached

    HTTP status codes:
      200 — healthy (no limit saturated)
      503 — saturated / degraded (APIM should stop routing new traffic here)
    """
    stats = admission_controller.stats
    is_degraded = stats["is_saturated"]

    # Emit a point-in-time metric snapshot to Application Insights on every probe
    try:
        from core.telemetry import track_metric
        track_metric("medvoice.active_analyses",   float(stats["active_analyses"]),   unit="sessions")
        track_metric("medvoice.active_websockets", float(stats["active_websockets"]), unit="sessions")
    except Exception:
        pass

    body = {
        "status":    "degraded" if is_degraded else "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version":   "M1.4",
        "admission": stats,
    }

    status_code = 503 if is_degraded else 200
    return Response(
        content=json.dumps(body),
        media_type="application/json",
        status_code=status_code,
    )


app.include_router(router)
app.include_router(patient_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999)