import time
import json
import logging
import asyncio
import secrets
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import uvicorn
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, APIRouter, Depends, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from core.config import Config
from model.speech_service import SpeechService
from model.llm_service import LLMService
from model.extract_features import ExtractFeature
from model.question_generator import QuestionGenerator
from model.refine_text import RefineText
from model.specialty_terms import correct_specialty_terms, SPECIALTIES
from model.patient_history import (
    get_patient_visits,
    format_history_for_prompt,
    check_history_usage,
    BusinessUnit,
    PatientHistoryError,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio Processing"])
patient_router = APIRouter(prefix="/patient", tags=["Patient History"])
app    = FastAPI(title="Medical Voice API")

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

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
async def admin_usage(limit: int = 500, _auth=Depends(_require_dashboard_auth)):
    limit = max(1, min(limit, 5000))
    return {"entries": _read_jsonl(USAGE_LOG, limit)}


@app.get("/admin/feedback", summary="Recent doctor feedback entries, newest first")
async def admin_feedback(limit: int = 500, _auth=Depends(_require_dashboard_auth)):
    limit = max(1, min(limit, 5000))
    return {"entries": _read_jsonl(FEEDBACK_LOG, limit)}


@app.get("/dashboard", summary="Analytics dashboard — usage + feedback, for internal review")
async def serve_dashboard(_auth=Depends(_require_dashboard_auth)):
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

@router.post("/refine-field", summary="Refine one short spoken answer for a single clinical-note field")
async def refine_field_answer(
    text:      str               = Form(..., description="Raw transcript of the doctor's spoken answer for one field"),
    language:  Language          = Form(Language.english, description="Language the answer was spoken in"),
    specialty: Optional[Specialty] = Form(None, description="Medical specialty — same fuzzy-term correction as the main pipeline"),
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    specialty_value = specialty.value if specialty else None
    try:
        refined_text = await asyncio.to_thread(
            RefineText.refining_transcription,
            raw_text=text,
            language=language.value,
            is_conversation=False,
            specialty=specialty_value,
        )
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
            refined_text = await asyncio.to_thread(
                LLMService.translate_to_eng,
                refined_text,
                Config.REFINE_API_KEY,
                is_conversation=False,
            )
        except Exception as exc:
            logger.error("[RefineField] Failed to translate field answer to English: %s", exc)
            # Leave the doctor with the refined Arabic text rather than nothing —
            # still better than losing the answer outright, and it's editable.

    return {"refined_text": refined_text}


# ═══════════════════════════════════════════════════════════════════════════════
# POST /audio/analyze — post-transcription pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/analyze", summary="Run refine → translate → extract → questions on a provided transcript")
async def analyze_transcript(
    request:    Request,
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

    async def event_stream():
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
                        LLMService.refine_ar_transcription_stream(text, Config.OPENROUTER_API_KEY, is_conversation=is_conversation, specialty=specialty_value)
                        if is_arabic else
                        LLMService.refine_en_transcription_stream(text, Config.OPENROUTER_API_KEY, is_conversation=is_conversation, specialty=specialty_value)
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
                    async for delta in LLMService.translate_to_eng_stream(
                        refined_text, Config.OPENROUTER_API_KEY, is_conversation=is_conversation
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

            # ── 3. Extract ──────────────────────────────────────────────
            t_extract = time.perf_counter()

            try:
                extracted_features, _ = await asyncio.to_thread(
                    ExtractFeature.extract,
                    end_text=final_text,
                    department=department_value,
                    patient_history_text=patient_history_text,
                    is_conversation=is_conversation,
                )
            except Exception as exc:
                log["status"] = "error"
                yield {"event": "error", "data": json.dumps({"step": "extraction", "detail": str(exc)})}
                return

            log["extracted_features"] = extracted_features
            # Audit trail: did the model actually report using the history we
            # gave it? (See ExtractFeature.extract's log warning for the case
            # where it didn't even return the key.) Cross-checked against what
            # actually shows up in icd10_codes/final_diagnosis (soft heuristic,
            # see check_history_usage's docstring) and surfaced to the
            # frontend, not just the server log.
            history_check = None
            if patient_history_text:
                history_check = check_history_usage(patient_history_visits, extracted_features)
                if history_check["flag"] == "claims_used_no_overlap":
                    logger.warning(
                        f"[Analyze] patient_code={patient_id}: model reported history_considered=Yes "
                        f"but no ICD-10 family overlap with the patient's history was found — "
                        f"history_notes={extracted_features.get('history_notes')!r}"
                    )
                elif history_check["flag"] == "overlap_but_not_flagged":
                    logger.warning(
                        f"[Analyze] patient_code={patient_id}: ICD-10 family overlap with patient history "
                        f"found ({history_check['overlap_families']}) but the model did not report "
                        f"history_considered=Yes"
                    )
            log["history_check"] = history_check

            # ── 4. Questions — derived directly from extracted_features, not a
            #      second LLM pass, so it can never disagree with what the
            #      extraction step already found (see QuestionGenerator).
            #      Computed here, BEFORE the yield below, so extraction +
            #      questions go out as ONE atomic SSE event rather than two
            #      separate ones. They used to be sent as two events, and each
            #      one, independently, has been observed to occasionally not
            #      reach the browser at all (almost certainly an intermediary
            #      — proxy/firewall — on this deployment mishandling one of
            #      the larger frames in a long-lived streamed response, not a
            #      backend bug: server-side computation and logs are correct
            #      every time this has been reported). Splitting them into two
            #      events just gave that failure mode two separate chances to
            #      hit instead of one, and led to exactly the inconsistent
            #      state seen in practice — features rendered, questions
            #      silently empty (or vice versa) — because each was an
            #      independent point of failure. Sending them together means
            #      there's only one handoff left for this critical data: it
            #      either fully arrives or fully doesn't, which is a much
            #      easier failure to detect and retry than a half-rendered UI. ──
            t_questions = time.perf_counter()
            raw_questions, _ = QuestionGenerator.generate(
                extracted_features=extracted_features,
                department=department_value,
            )
            log["questions"] = raw_questions
            total_sec        = round(time.perf_counter() - pipeline_start, 3)
            extraction_sec           = round(time.perf_counter() - t_extract, 3)
            question_generation_sec  = round(time.perf_counter() - t_questions, 3)
            log["extraction_sec"]          = extraction_sec
            log["question_generation_sec"] = question_generation_sec

            extraction_payload = {
                "extracted_features":      extracted_features,
                "extraction_sec":          extraction_sec,
                "department":              department_value,
                "patient_story":           patient_story,
                "history_used":            bool(patient_history_text),
                "history_visit_count":     len(patient_history_visits),
                "history_check":           history_check,
                "history_error":           history_error,
                "questions":               raw_questions,
                "question_generation_sec": question_generation_sec,
                "total_sec":               total_sec,
            }
            logger.info(
                f"[Analyze] Emitting combined 'extraction' SSE event "
                f"({len(extracted_features)} field(s), {len(raw_questions)} question(s), "
                f"~{len(json.dumps(extraction_payload))} bytes)"
            )
            yield {"event": "extraction", "data": json.dumps(extraction_payload)}

        finally:
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


app.include_router(router)
app.include_router(patient_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7777)