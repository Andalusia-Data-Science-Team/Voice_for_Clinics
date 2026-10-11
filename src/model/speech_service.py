import os
import asyncio
import logging
import shutil
import time
from typing import Optional, Tuple, Dict, Any, AsyncIterator

from mistralai import Mistral
from mistralai.models import (
    AudioFormat,
    RealtimeTranscriptionError,
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
)
from mistralai.extra.realtime import UnknownRealtimeEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOXTRAL_MODEL = "voxtral-mini-transcribe-realtime-2602"
VOXTRAL_AUDIO_FORMAT = AudioFormat(encoding="pcm_s16le", sample_rate=16000)
_CHUNK_SIZE = 4096

_mistral_client: Optional[Mistral] = None


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""


def _get_client(api_key: str) -> Mistral:
    global _mistral_client
    if _mistral_client is None:
        _mistral_client = Mistral(api_key=api_key)
        logger.debug("[SpeechService] Mistral client instantiated")
    return _mistral_client


# ---------------------------------------------------------------------------
# Audio source generators
# ---------------------------------------------------------------------------

async def _ffmpeg_file_stream(input_path: str, chunk_size: int = _CHUNK_SIZE) -> AsyncIterator[bytes]:
    """Convert a file to pcm_s16le via ffmpeg stdout pipe — no temp file."""
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("ffmpeg not found on PATH.")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        while True:
            chunk = await proc.stdout.read(chunk_size)
            if not chunk:
                break
            yield chunk
            
            
            
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        _, stderr = await asyncio.gather(asyncio.sleep(0), proc.stderr.read())
        if proc.returncode not in (0, -9):
            raise TranscriptionError(
                f"ffmpeg failed (exit {proc.returncode}): {stderr.decode(errors='replace')}"
            )


async def _ffmpeg_stdin_stream(
    raw_pcm_iter: AsyncIterator[bytes],
    chunk_size: int = _CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """
    Pipe a live stream of browser PCM (float32, 16kHz, mono) through ffmpeg,
    converting it to pcm_s16le that Voxtral expects.

    Browser AudioContext captures float32 at 16kHz mono — ffmpeg converts
    f32le → s16le on the fly.
    """
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("ffmpeg not found on PATH.")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "f32le",       # browser sends float32 little-endian
        "-ar", "16000",      # sample rate matches AudioContext
        "-ac", "1",          # mono
        "-i", "pipe:0",      # read from stdin
        "-f", "s16le",       # output format Voxtral needs
        "-ar", "16000",
        "-ac", "1",
        "pipe:1",            # write to stdout
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def feed_stdin():
        """Write incoming PCM chunks into ffmpeg stdin."""
        try:
            async for chunk in raw_pcm_iter:
                proc.stdin.write(chunk)
                await proc.stdin.drain()
        except Exception as e:
            logger.warning("[ffmpeg_stdin] Feed error: %s", e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    # Start feeding stdin in the background
    feed_task = asyncio.create_task(feed_stdin())

    try:
        while True:
            chunk = await proc.stdout.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        feed_task.cancel()
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        _, stderr = await asyncio.gather(asyncio.sleep(0), proc.stderr.read())
        if proc.returncode not in (0, -9):
            raise TranscriptionError(
                f"ffmpeg stdin failed (exit {proc.returncode}): {stderr.decode(errors='replace')}"
            )


async def _plain_file_stream(file_path: str, chunk_size: int = _CHUNK_SIZE) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    with open(file_path, "rb") as fh:
        while True:
            chunk = await loop.run_in_executor(None, fh.read, chunk_size)
            if not chunk:
                break
            yield chunk


# ---------------------------------------------------------------------------
# Voxtral wrappers
# ---------------------------------------------------------------------------

async def _transcribe_collect(audio_stream: AsyncIterator[bytes], api_key: str) -> str:
    """Collect all deltas → return full string. Used for Arabic (needs full text before translate)."""
    client = _get_client(api_key)
    parts: list[str] = []
    async for event in client.audio.realtime.transcribe_stream(
        audio_stream=audio_stream,
        model=VOXTRAL_MODEL,
        audio_format=VOXTRAL_AUDIO_FORMAT,
        target_streaming_delay_ms=500
    ):
        if isinstance(event, TranscriptionStreamTextDelta):
            parts.append(event.text)
        elif isinstance(event, TranscriptionStreamDone):
            break
        elif isinstance(event, RealtimeTranscriptionError):
            raise TranscriptionError(f"Voxtral error: {event}")
        elif isinstance(event, UnknownRealtimeEvent):
            logger.warning("[Voxtral] Unknown event: %s", event)
    return "".join(parts)


async def _transcribe_yield(
    audio_stream: AsyncIterator[bytes], api_key: str
) -> AsyncIterator[str]:
    """Yield each delta as it arrives. Used for English live streaming."""
    client = _get_client(api_key)
    async for event in client.audio.realtime.transcribe_stream(
        audio_stream=audio_stream,
        model=VOXTRAL_MODEL,
        audio_format=VOXTRAL_AUDIO_FORMAT,
        target_streaming_delay_ms=500
    ):
        if isinstance(event, TranscriptionStreamTextDelta):
            yield event.text
        elif isinstance(event, TranscriptionStreamDone):
            break
        elif isinstance(event, RealtimeTranscriptionError):
            raise TranscriptionError(f"Voxtral error: {event}")
        elif isinstance(event, UnknownRealtimeEvent):
            logger.warning("[Voxtral] Unknown event: %s", event)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SpeechService:
    """
    Speech recognition via Mistral Voxtral real-time transcription.

    Entry points:
      transcribe_audio()            – batch, from file  (Arabic path)
      transcribe_audio_stream()     – streaming deltas, from file  (English post-recording)
      transcribe_live_stream()      – streaming deltas, from live WebSocket bytes  (English live)
    """

    # ------------------------------------------------------------------
    # Batch from file — Arabic
    # ------------------------------------------------------------------
    @staticmethod
    async def transcribe_audio(
        audio_file_path: str,
        api_key: str,
        language: str = "en",
        preprocess: bool = True,
        model: str = VOXTRAL_MODEL,
        timeout: int = 300,
        return_meta: bool = False,
    ) -> "str | Tuple[str, Dict[str, Any]]":
        if not api_key:
            raise ValueError("Missing Mistral API key.")
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        try:
            stream = _ffmpeg_file_stream(audio_file_path) if preprocess else _plain_file_stream(audio_file_path)
            start = time.time()
            text = await _transcribe_collect(stream, api_key)
            elapsed = time.time() - start
            if not text.strip():
                raise TranscriptionError("Voxtral returned empty transcript.")
            logger.info("[SpeechService] Batch done in %.2fs: %d chars", elapsed, len(text))
            if return_meta:
                return text, {"model": VOXTRAL_MODEL, "transcription_time": round(elapsed, 2)}
            return text
        except (FileNotFoundError, ValueError, TranscriptionError):
            raise
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Streaming deltas from file — English, post-recording fallback
    # ------------------------------------------------------------------
    @staticmethod
    async def transcribe_audio_stream(
        audio_file_path: str,
        api_key: str,
        preprocess: bool = True,
    ) -> AsyncIterator[str]:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        stream = _ffmpeg_file_stream(audio_file_path) if preprocess else _plain_file_stream(audio_file_path)
        async for delta in _transcribe_yield(stream, api_key):
            yield delta

    # ------------------------------------------------------------------
    # Streaming deltas from live WebSocket bytes — English, real-time
    # ------------------------------------------------------------------
    @staticmethod
    async def transcribe_live_stream(
        raw_pcm_iter: AsyncIterator[bytes],
        api_key: str,
    ) -> AsyncIterator[str]:
        """
        Accept a live async iterator of float32 PCM bytes (from browser AudioContext),
        convert via ffmpeg stdin pipe, stream to Voxtral, yield text deltas in real time.
        """
        audio_stream = _ffmpeg_stdin_stream(raw_pcm_iter)
        async for delta in _transcribe_yield(audio_stream, api_key):
            yield delta

    @staticmethod
    def transcribe_audio_sync(audio_file_path: str, api_key: str, **kwargs):
        return asyncio.run(SpeechService.transcribe_audio(audio_file_path, api_key, **kwargs))