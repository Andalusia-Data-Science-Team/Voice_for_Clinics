import os
import time
import json
import logging
import asyncio
import shutil
from enum import Enum
from pathlib import Path
from datetime import datetime

import uvicorn
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, APIRouter, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.config import Config
from model.speech_service import SpeechService, TranscriptionError
from model.llm_service import LLMService
from model.translation import Translate
from model.extract_features import ExtractFeature
from model.question_generator import QuestionGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio Processing"])
app = FastAPI(title="Medical Voice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1110"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Uploads directory (src/uploads/) ──────────────────────────────────────────
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
    summary="Transcribe audio, extract features, and generate questions (streamed)",
)
async def process_audio(
    file_path: str = Form(..., description="Absolute or relative path to the audio file on the server"),
    language: Language = Form(Language.arabic, description="Audio language"),
    mode: Mode = Form(Mode.doctor, description="Doctor Mode (single clinician) or Conversation Mode (doctor-patient)"),
):
    pipeline_start = time.perf_counter()
    is_arabic = language == Language.arabic
    is_conversation = mode == Mode.conversation

    # ── Validate path upfront — fail fast before opening the stream ───────
    src_path = Path(file_path)
    if not src_path.is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {file_path!r}")

    # ── Copy audio into a fresh timestamped session folder ────────────────
    session_dir = _session_dir(src_path.name)
    audio_path = session_dir / src_path.name
    shutil.copy2(src_path, audio_path)
    logger.info("Audio copied to session: %s", audio_path)

    async def event_stream():
        results: dict = {
            "filename": src_path.name,
            "language": language.value,
            "mode": mode.value,
            "session_dir": str(session_dir),
        }

        try:
            # ── 1. Transcribe ──────────────────────────────────────────────
            logger.info("Transcribing: %s  language=%s  mode=%s", audio_path, language, mode)
            t0 = time.perf_counter()
            try:
                raw_transcript = await asyncio.to_thread(
                    SpeechService.transcribe_audio,
                    audio_file_path=str(audio_path),
                    api_key=Config.FIREWORKS_API_KEY,
                    language=language.value,
                    preprocess=False,
                )
            except (TranscriptionError, FileNotFoundError, ValueError) as exc:
                yield {"event": "error", "data": json.dumps({"step": "transcription", "detail": str(exc)})}
                return
            transcription_sec = round(time.perf_counter() - t0, 3)

            # ── 2. Refine ──────────────────────────────────────────────────
            t0 = time.perf_counter()
            if is_arabic:
                refined_text = LLMService.refine_ar_transcription(
                    raw_transcript, Config.FIREWORKS_API_KEY, is_conversation=is_conversation
                )
            else:
                refined_text = LLMService.refine_en_transcription(
                    raw_transcript, Config.FIREWORKS_API_KEY, is_conversation=is_conversation
                )
            refinement_sec = round(time.perf_counter() - t0, 3)

            # ── 3. Translate (Arabic → English) ───────────────────────────
            translation_sec = None
            if is_arabic:
                t0 = time.perf_counter()
                final_text = Translate.translate(refined_text, is_conversation=is_conversation)
                translation_sec = round(time.perf_counter() - t0, 3)
            else:
                final_text = refined_text

            transcription_payload = {
                "final_text": final_text,
                "transcription_sec": transcription_sec,
                "refinement_sec": refinement_sec,
                "translation_sec": translation_sec,
            }
            results["transcription"] = transcription_payload
            yield {"event": "transcription", "data": json.dumps(transcription_payload)}

            # ── 4 & 5. Extract features + Generate questions (parallel) ────
            t0 = time.perf_counter()
            try:
                (extracted_features, _), (raw_questions, _) = await asyncio.gather(
                    asyncio.to_thread(
                        ExtractFeature.extract,
                        end_text=final_text,
                        is_conversation=is_conversation,
                    ),
                    asyncio.to_thread(
                        QuestionGenerator.generate,
                        translated_text=final_text,
                        is_conversation=is_conversation,
                    ),
                )
            except Exception as exc:
                yield {"event": "error", "data": json.dumps({"step": "extraction/questions", "detail": str(exc)})}
                return

            parallel_sec = round(time.perf_counter() - t0, 3)
            total_sec = round(time.perf_counter() - pipeline_start, 3)

            extraction_payload = {"extracted_features": extracted_features, "extraction_sec": parallel_sec}
            questions_payload = {
                "questions": raw_questions,
                "question_generation_sec": parallel_sec,
                "total_sec": total_sec,
            }

            results["extraction"] = extraction_payload
            results["questions"] = questions_payload
            results["total_sec"] = total_sec

            yield {"event": "extraction", "data": json.dumps(extraction_payload)}
            yield {"event": "questions", "data": json.dumps(questions_payload)}

        except Exception as exc:
            logger.exception("Unexpected error in /audio/process")
            results["error"] = str(exc)
            yield {"event": "error", "data": json.dumps({"step": "unknown", "detail": str(exc)})}

        finally:
            # ── Save results JSON into the session folder ──────────────────
            try:
                results_path = session_dir / "results.json"
                results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                logger.info("Results saved: %s", results_path)
            except Exception as save_err:
                logger.warning("Failed to save results: %s", save_err)

    return EventSourceResponse(event_stream())


app.include_router(router)

if __name__ == "__main__":
    print("ENV KEY:", os.environ.get("FIREWORKS_API_KEY"))
    print("FIREWORKS_API_KEY:", Config.FIREWORKS_API_KEY)
    uvicorn.run("app:app", host="0.0.0.0", port=9999)