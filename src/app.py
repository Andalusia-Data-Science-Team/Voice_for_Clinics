import time
import json
import logging
import asyncio
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import uvicorn
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, APIRouter, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.config import Config
from model.speech_service import SpeechService
from model.llm_service import LLMService
from model.extract_features import ExtractFeature
from model.question_generator import QuestionGenerator


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["Audio Processing"])
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


def save_usage_log(entry: dict) -> None:
    try:
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("[UsageLog] Saved entry for %s", entry.get("ip"))
    except Exception as e:
        logger.warning("[UsageLog] Failed to write log: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    arabic  = "ar"
    english = "en"


class Mode(str, Enum):
    doctor       = "doctor"
    conversation = "conversation"


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
    language = "en"
    hotwords = None
    try:
        init_text = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        init_data = json.loads(init_text)
        input_sample_rate = int(init_data.get("sampleRate", 16000))
        language = str(init_data.get("language", "en")).lower()
        hotwords = init_data.get("hotwords") or None
        logger.info("[WS] Browser sample rate: %d Hz, language: %s", input_sample_rate, language)
    except asyncio.TimeoutError:
        logger.warning("[WS] No init message within 5 s, assuming en/16000 Hz")
    except Exception as e:
        logger.warning("[WS] Bad init message: %s — assuming en/16000 Hz", e)

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
# POST /audio/analyze — post-transcription pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/analyze", summary="Run refine → translate → extract → questions on a provided transcript")
async def analyze_transcript(
    request:  Request,
    text:     str      = Form(...,              description="Raw transcript from live transcription"),
    language: Language = Form(Language.english, description="Audio language"),
    mode:     Mode     = Form(Mode.doctor,      description="Doctor or Conversation mode"),
):
    pipeline_start  = time.perf_counter()
    is_arabic       = language == Language.arabic
    is_conversation = mode == Mode.conversation

    async def event_stream():
        # Accumulated outputs — populated as each step completes
        log: dict = {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "ip":                 _get_client_ip(request),
            "user_agent":         request.headers.get("user-agent", "unknown"),
            "language":           language.value,
            "mode":               mode.value,
            "status":             "ok",
            "total_sec":          None,
            "transcript":         text,
            "refined_text":       None,
            "translated_text":    None,
            "extracted_features": None,
            "questions":          None,
        }

        try:
            # ── 1. Refine ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            refined_parts: list[str] = []
            try:
                refine_stream = (
                    LLMService.refine_ar_transcription_stream(text, Config.OPENROUTER_API_KEY, is_conversation=is_conversation)
                    if is_arabic else
                    LLMService.refine_en_transcription_stream(text, Config.OPENROUTER_API_KEY, is_conversation=is_conversation)
                )
                async for delta in refine_stream:
                    refined_parts.append(delta)
                    yield {"event": "refinement_delta", "data": json.dumps({"delta": delta})}
            except Exception as exc:
                log["status"] = "error"
                yield {"event": "error", "data": json.dumps({"step": "refinement", "detail": str(exc)})}
                return

            refined_text      = "".join(refined_parts)
            log["refined_text"] = refined_text
            yield {"event": "refinement_done", "data": json.dumps({
                "refined_text":   refined_text,
                "refinement_sec": round(time.perf_counter() - t0, 3),
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
                log["translated_text"] = final_text
                yield {"event": "translation_done", "data": json.dumps({
                    "final_text":      final_text,
                    "translation_sec": round(time.perf_counter() - t0, 3),
                })}
            else:
                final_text = refined_text

            # ── 3 & 4. Extract + Questions (concurrent) ───────────────────
            t_extract   = time.perf_counter()
            t_questions = time.perf_counter()

            extract_task = asyncio.create_task(
                asyncio.to_thread(ExtractFeature.extract, end_text=final_text, is_conversation=is_conversation)
            )
            questions_task = asyncio.create_task(
                asyncio.to_thread(QuestionGenerator.generate, translated_text=final_text, is_conversation=is_conversation)
            )

            try:
                extracted_features, _ = await extract_task
            except Exception as exc:
                questions_task.cancel()
                log["status"] = "error"
                yield {"event": "error", "data": json.dumps({"step": "extraction", "detail": str(exc)})}
                return

            log["extracted_features"] = extracted_features
            yield {"event": "extraction", "data": json.dumps({
                "extracted_features": extracted_features,
                "extraction_sec":     round(time.perf_counter() - t_extract, 3),
            })}

            try:
                raw_questions, _ = await questions_task
            except Exception as exc:
                log["status"] = "error"
                yield {"event": "error", "data": json.dumps({"step": "questions", "detail": str(exc)})}
                return

            log["questions"] = raw_questions
            total_sec        = round(time.perf_counter() - pipeline_start, 3)
            yield {"event": "questions", "data": json.dumps({
                "questions":               raw_questions,
                "question_generation_sec": round(time.perf_counter() - t_questions, 3),
                "total_sec":               total_sec,
            })}

        finally:
            log["total_sec"] = round(time.perf_counter() - pipeline_start, 3)
            await asyncio.to_thread(save_usage_log, log)

    return EventSourceResponse(event_stream())


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7777)