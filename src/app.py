import os
import time
import json
import logging
import tempfile
import asyncio
from enum import Enum

import uvicorn
from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, APIRouter, File, Form, UploadFile

from core.config import Config
from model.speech_service import SpeechService, TranscriptionError
from model.llm_service import LLMService
from model.translation import Translate
from model.extract_features import ExtractFeature
from model.question_generator import QuestionGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio Processing"])
app = FastAPI(title="Medical Voice API")


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
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, ogg, …)"),
    language: Language = Form(Language.arabic, description="Audio language"),
    mode: Mode = Form(Mode.doctor, description="Doctor Mode (single clinician) or Conversation Mode (doctor-patient)"),
):
    pipeline_start = time.perf_counter()
    is_arabic = language == Language.arabic
    is_conversation = mode == Mode.conversation

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    async def event_stream():
        try:
            # ── 1. Transcribe ──────────────────────────────────────────────
            logger.info("Transcribing: %s  language=%s  mode=%s", file.filename, language, mode)
            t0 = time.perf_counter()
            try:
                raw_transcript = await asyncio.to_thread(
                    SpeechService.transcribe_audio,
                    audio_file_path=tmp_path,
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

            yield {
                "event": "transcription",
                "data": json.dumps({
                    "final_text": final_text,
                    "transcription_sec": transcription_sec,
                    "refinement_sec": refinement_sec,
                    "translation_sec": translation_sec,
                }),
            }

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

            yield {
                "event": "extraction",
                "data": json.dumps({"extracted_features": extracted_features, "extraction_sec": parallel_sec}),
            }
            yield {
                "event": "questions",
                "data": json.dumps({
                    "questions": raw_questions,
                    "question_generation_sec": parallel_sec,
                    "total_sec": round(time.perf_counter() - pipeline_start, 3),
                }),
            }

        except Exception as exc:
            logger.exception("Unexpected error in /audio/process")
            yield {"event": "error", "data": json.dumps({"step": "unknown", "detail": str(exc)})}

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return EventSourceResponse(event_stream())


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999)