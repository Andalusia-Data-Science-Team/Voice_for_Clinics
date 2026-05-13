import os
import time
import json
import logging
import asyncio
from enum import Enum
from pathlib import Path
from datetime import datetime

import uvicorn
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, APIRouter, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.config import Config
from model.speech_service import SpeechService, TranscriptionError
from model.llm_service import LLMService
from model.extract_features import ExtractFeature
from model.question_generator import QuestionGenerator

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio Processing"])
app = FastAPI(title="Medical Voice API")

# Mount static files — add this after app is created
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Serve index.html at root
@app.get("/")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1110"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
        )

# ── Uploads directory ─────────────────────────────────────────────────────────
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


# --------------- Helpers --------------- #

def _session_dir(filename: str) -> Path:
    """Create a timestamped subfolder per request: uploads/20250101_123045_recording/"""
    stem = Path(filename).stem if filename else "audio"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = UPLOADS_DIR / f"{timestamp}_{stem}"
    session.mkdir(parents=True, exist_ok=True)
    return session


# --------------- Enums --------------- #

class Language(str, Enum):
    arabic = "ar"
    english = "en"


class Mode(str, Enum):
    doctor = "doctor"               # is_conversation=False
    conversation = "conversation"   # is_conversation=True


# --------------- Endpoint --------------- #

@router.post(
    "/process",
    summary="Transcribe recorded audio and stream all pipeline results",
)
async def process_audio(
    file: UploadFile = File(..., description="Recorded audio file (webm, wav, mp3, ogg, m4a, …)"),
    language: Language = Form(Language.arabic, description="Audio language"),
    mode: Mode = Form(Mode.doctor, description="Doctor Mode (single clinician) or Conversation Mode (doctor-patient)"),
):
    pipeline_start = time.perf_counter()
    is_arabic = language == Language.arabic
    is_conversation = mode == Mode.conversation

    # ── Save uploaded recording into a timestamped session folder ─────────
    original_name = file.filename or f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
    session_dir = _session_dir(original_name)
    audio_path = session_dir / original_name

    contents = await file.read()
    audio_path.write_bytes(contents)
    logger.info("Audio saved: %s (%d bytes)", audio_path, len(contents))

    async def event_stream():
        results: dict = {
            "filename": original_name,
            "language": language.value,
            "mode": mode.value,
            "session_dir": str(session_dir),
        }

        try:
            # ── 1. Transcribe — stream deltas to client ────────────────────
            logger.info("Transcribing: %s  language=%s  mode=%s", audio_path, language, mode)
            t0 = time.perf_counter()
            transcript_parts: list[str] = []

            try:
                async for delta in SpeechService.transcribe_audio_stream(
                    audio_file_path=str(audio_path),
                    api_key=Config.MISTRAL_API_KEY,
                    preprocess=True,
                ):
                    transcript_parts.append(delta)
                    yield {
                        "event": "transcription_delta",
                        "data": json.dumps({"delta": delta}),
                    }

            except (TranscriptionError, FileNotFoundError, ValueError) as exc:
                yield {"event": "error", "data": json.dumps({"step": "transcription", "detail": str(exc)})}
                return

            raw_transcript = "".join(transcript_parts)
            transcription_sec = round(time.perf_counter() - t0, 3)

            # Signal transcription is complete
            yield {
                "event": "transcription_done",
                "data": json.dumps({
                    "raw_transcript": raw_transcript,
                    "transcription_sec": transcription_sec,
                }),
            }
            logger.info("Transcription done in %.3fs: %d chars", transcription_sec, len(raw_transcript))

            # ── 2. Refine — stream deltas to client ───────────────────────
            t0 = time.perf_counter()
            refined_parts: list[str] = []

            try:
                if is_arabic:
                    refine_stream = LLMService.refine_ar_transcription_stream(
                        raw_transcript, Config.OPENROUTER_API_KEY, is_conversation=is_conversation
                    )
                else:
                    refine_stream = LLMService.refine_en_transcription_stream(
                        raw_transcript, Config.OPENROUTER_API_KEY, is_conversation=is_conversation
                    )

                async for delta in refine_stream:
                    refined_parts.append(delta)
                    yield {
                        "event": "refinement_delta",
                        "data": json.dumps({"delta": delta}),
                    }

            except Exception as exc:
                yield {"event": "error", "data": json.dumps({"step": "refinement", "detail": str(exc)})}
                return

            refined_text = "".join(refined_parts)
            refinement_sec = round(time.perf_counter() - t0, 3)

            yield {
                "event": "refinement_done",
                "data": json.dumps({
                    "refined_text": refined_text,
                    "refinement_sec": refinement_sec,
                }),
            }

            # ── 3. Translate (Arabic → English) — stream deltas ───────
            translation_sec = None
            if is_arabic:
                t0 = time.perf_counter()
                translated_parts: list[str] = []

                try:
                    async for delta in LLMService.translate_to_eng_stream(
                        refined_text, Config.OPENROUTER_API_KEY, is_conversation=is_conversation
                    ):
                        translated_parts.append(delta)
                        yield {
                            "event": "translation_delta",
                            "data": json.dumps({"delta": delta}),
                        }
                except Exception as exc:
                    yield {"event": "error", "data": json.dumps({"step": "translation", "detail": str(exc)})}
                    return

                final_text = "".join(translated_parts)
                translation_sec = round(time.perf_counter() - t0, 3)

                yield {
                    "event": "translation_done",
                    "data": json.dumps({
                        "final_text": final_text,
                        "translation_sec": translation_sec,
                    }),
                }
            else:
                final_text = refined_text

            # Save to results — no longer yielded as a combined event
            results["transcription"] = {
                "raw_transcript": raw_transcript,
                "refined_text": refined_text,
                "final_text": final_text,
                "transcription_sec": transcription_sec,
                "refinement_sec": refinement_sec,
                "translation_sec": translation_sec,
            }

            # ── 4. Extract features ────────────────────────────────────
            t0 = time.perf_counter()
            try:
                extracted_features, _ = await asyncio.to_thread(
                    ExtractFeature.extract,
                    end_text=final_text,
                    is_conversation=is_conversation,
                )
            except Exception as exc:
                yield {"event": "error", "data": json.dumps({"step": "extraction", "detail": str(exc)})}
                return

            extraction_sec = round(time.perf_counter() - t0, 3)

            extraction_payload = {
                "extracted_features": extracted_features,
                "extraction_sec": extraction_sec,
            }
            results["extraction"] = extraction_payload
            yield {"event": "extraction", "data": json.dumps(extraction_payload)}

            # ── 5. Generate questions (after extraction completes) ─────
            t0 = time.perf_counter()
            try:
                raw_questions, _ = await asyncio.to_thread(
                    QuestionGenerator.generate,
                    translated_text=final_text,
                    is_conversation=is_conversation,
                )
            except Exception as exc:
                yield {"event": "error", "data": json.dumps({"step": "questions", "detail": str(exc)})}
                return

            question_sec = round(time.perf_counter() - t0, 3)
            total_sec    = round(time.perf_counter() - pipeline_start, 3)

            questions_payload = {
                "questions": raw_questions,
                "question_generation_sec": question_sec,
                "total_sec": total_sec,
            }
            results["questions"]  = questions_payload
            results["total_sec"]  = total_sec
            yield {"event": "questions", "data": json.dumps(questions_payload)}

        except Exception as exc:
            logger.exception("Unexpected error in /audio/process")
            results["error"] = str(exc)
            yield {"event": "error", "data": json.dumps({"step": "unknown", "detail": str(exc)})}

        finally:
            try:
                results_path = session_dir / "results.json"
                results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                logger.info("Results saved: %s", results_path)
            except Exception as save_err:
                logger.warning("Failed to save results: %s", save_err)

    return EventSourceResponse(event_stream())


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999)