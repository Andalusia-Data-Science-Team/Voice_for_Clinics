import os
import logging
from typing import Tuple, Dict, Any
import requests

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""


class SpeechService:
    """Service for speech recognition with optional audio preprocessing."""

    FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    TRANSCRIBE_ENDPOINT = f"{FIREWORKS_BASE_URL}/inference/v1/audio/transcriptions"

    @staticmethod
    def transcribe_audio(
        audio_file_path: str,
        api_key: str,
        language: str = "en",
        preprocess: bool = True,
        model: str = "whisper-v3",
        timeout: int = 300,
        return_meta: bool = False,
    ) -> str | Tuple[str, Dict[str, Any]]:
        """
        Transcribe an audio file using Fireworks Whisper.

        Args:
            audio_file_path: Path to the audio file.
            api_key: Fireworks API key.
            language: ISO language code used by Whisper (e.g., "en", "ar").
            preprocess: Whether to apply audio preprocessing (uses audio_preprocessing module if present).
            model: Fireworks Whisper model name. Default: "whisper-v3".
            timeout: HTTP request timeout in seconds.
            return_meta: If True, returns (text, meta_dict) instead of just text.

        Returns:
            Transcribed text (or (text, metadata) if return_meta=True).

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            ValueError: If inputs are invalid.
            TranscriptionError: For HTTP/JSON or service errors.
        """
        if not api_key:
            raise ValueError("Missing Fireworks API key.")
        if not audio_file_path:
            raise ValueError("audio_file_path must be provided.")

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        processed_file_path = audio_file_path
        temp_file_created = False

        # Optional preprocessing
        if preprocess:
            try:
                from .audio_preprocessing import AudioPreprocessingService  # optional
                processed_file_path = AudioPreprocessingService.preprocess_audio(audio_file_path)
                temp_file_created = processed_file_path != audio_file_path
                logger.info("Audio preprocessing applied: %s → %s", audio_file_path, processed_file_path)
            except Exception as e:
                # Preprocessing is optional; you can choose to fail or continue.
                # Here we *fail fast* to keep behavior explicit.
                raise TranscriptionError(f"Audio preprocessing failed: {e}") from e

        try:
            headers = {"Authorization": f"Bearer {api_key}"}

            # Build multipart form-data
            data = {"model": model}
            if language:
                data["language"] = language

            logger.info("Starting transcription: file=%s, model=%s, language=%s", processed_file_path, model, language)

            with open(processed_file_path, "rb") as f:
                files = {"file": (os.path.basename(processed_file_path), f, "application/octet-stream")}
                resp = requests.post(
                    SpeechService.TRANSCRIBE_ENDPOINT,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=timeout,
                )

            if resp.status_code >= 400:
                # Try to surface server error details
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise TranscriptionError(
                    f"Fireworks transcription error [{resp.status_code}]: {detail}"
                )

            # Parse JSON
            try:
                payload = resp.json()
            except Exception as e:
                raise TranscriptionError(f"Invalid JSON response from Fireworks: {e}") from e

            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise TranscriptionError(f"Fireworks response missing 'text': {payload}")

            logger.info("Transcription completed successfully: %d characters", len(text))

            if return_meta:
                meta = {
                    "model": model,
                    "language": language,
                    "endpoint": SpeechService.TRANSCRIBE_ENDPOINT,
                    "status_code": resp.status_code,
                }
                return text, meta
            return text

        except (FileNotFoundError, ValueError):
            # bubble up these explicitly
            raise
        except requests.Timeout as e:
            raise TranscriptionError(f"Transcription timed out after {timeout}s") from e
        except requests.RequestException as e:
            raise TranscriptionError(f"HTTP error during transcription: {e}") from e
        except Exception as e:
            raise TranscriptionError(f"Audio transcription failed: {e}") from e
        finally:
            # Clean up temporary processed file if we created one
            try:
                if temp_file_created and processed_file_path != audio_file_path and os.path.exists(processed_file_path):
                    os.remove(processed_file_path)
                    logger.debug("Removed temporary preprocessed file: %s", processed_file_path)
            except Exception as cleanup_err:
                logger.warning("Failed to remove temporary file %s: %s", processed_file_path, cleanup_err)
