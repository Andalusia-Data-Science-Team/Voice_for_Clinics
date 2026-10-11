import os
import json
import asyncio
import logging
import shutil
import time
import urllib.parse
from typing import Optional, Tuple, Dict, Any, AsyncIterator

import httpx
import websockets
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

MUNSIT_WS_URL = "wss://api.munsit.com/api/v1/listen"
MUNSIT_MODEL = "munsit"          # or "munsit-en-ar" for code-switching sessions
MUNSIT_IDLE_TIMEOUT_S = 12       # server closes at 12s with no audio/KeepAlive
MUNSIT_KEEPALIVE_INTERVAL_S = 8  # send KeepAlive comfortably before the 12s cutoff

# ---------------------------------------------------------------------------
# Munsit Text-to-Speech — session recording-consent disclaimer
# ---------------------------------------------------------------------------
MUNSIT_TTS_API_BASE   = "https://api.munsit.com/api/v1"
MUNSIT_TTS_MODEL      = "faseeh-v1-preview"   # Faseeh — Munsit's TTS model
DISCLAIMER_VOICE_NAME = "Lina"
DISCLAIMER_DIALECT    = "fusha"               # Modern Standard Arabic
DISCLAIMER_TEXT       = ".نود إعلامكم بأن هذه الجلسة يتم تسجيلها واستخدامها لأغراض التوثيق الطبي"

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
    input_sample_rate: int = 16000,
) -> AsyncIterator[bytes]:
    """
    Pipe a live stream of browser PCM (float32, any sample rate, mono) through ffmpeg,
    converting it to pcm_s16le at 16kHz — the format both Voxtral and Munsit expect.

    The browser's AudioContext may capture at 44100 Hz or 48000 Hz — the
    caller passes the actual rate so ffmpeg can resample correctly.
    """
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("ffmpeg not found on PATH.")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "f32le",                     # browser sends float32 little-endian
        "-ar", str(input_sample_rate),     # actual browser sample rate (e.g. 44100)
        "-ac", "1",                        # mono
        "-i", "pipe:0",                    # read from stdin
        "-f", "s16le",                     # linear16 output — Voxtral and Munsit both want this
        "-ar", "16000",                    # resample to 16 kHz
        "-ac", "1",
        "pipe:1",                          # write to stdout
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
# Voxtral wrappers (English)
# ---------------------------------------------------------------------------

async def _transcribe_collect(audio_stream: AsyncIterator[bytes], api_key: str) -> str:
    """Collect all deltas → return full string."""
    client = _get_client(api_key)
    parts: list[str] = []
    async for event in client.audio.realtime.transcribe_stream(
        audio_stream=audio_stream,
        model=VOXTRAL_MODEL,
        audio_format=VOXTRAL_AUDIO_FORMAT,
        target_streaming_delay_ms=500,
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
    """Yield each delta as it arrives."""
    client = _get_client(api_key)
    async for event in client.audio.realtime.transcribe_stream(
        audio_stream=audio_stream,
        model=VOXTRAL_MODEL,
        audio_format=VOXTRAL_AUDIO_FORMAT,
        target_streaming_delay_ms=500,
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
# Munsit wrapper (Arabic — /api/v1/listen)
# ---------------------------------------------------------------------------

def _build_munsit_url(api_key: str, hotwords: Optional[str] = None, model: str = MUNSIT_MODEL) -> str:
    params = {
        "encoding": "linear16",
        "sample_rate": "16000",
        "model": model,
        "language": "ar",
        "interim_results": "true",
        "smart_turn": "true",
    }
    if hotwords:
        params["hotwords"] = hotwords
    # api_key goes in the query string — the browser can't reach Munsit anyway
    # since this call happens server-side, so it never touches client logs.
    return f"{MUNSIT_WS_URL}?api_key={urllib.parse.quote(api_key)}&{urllib.parse.urlencode(params)}"


async def _transcribe_munsit_yield(
    audio_stream: AsyncIterator[bytes],
    api_key: str,
    hotwords: Optional[str] = None,
    model: str = MUNSIT_MODEL,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream 16kHz linear16 PCM to Munsit and yield normalized events:

      {"type": "interim",  "text": "..."}                      — provisional, may be revised
      {"type": "final",    "text": "...", "speech_final": bool} — committed text for this turn
      {"type": "utterance_end"}                                 — caller genuinely stopped talking
      {"type": "done", "billed_seconds": float}
      {"type": "error", "detail": "...", "recoverable": bool}

    speech_final=False on a "final" event means a forced 60s split mid-speech —
    the text is stable but the turn is still open (per Munsit docs).

    Note: the sender's KeepAlive loop below runs on a fixed 8s cadence regardless
    of whether audio chunks are arriving — including during a client-side pause,
    where no audio is sent at all — so the socket survives an arbitrarily long
    pause without hitting Munsit's 12s idle timeout.
    """
    url = _build_munsit_url(api_key, hotwords=hotwords, model=model)

    try:
        async with websockets.connect(url, max_size=None, open_timeout=10) as ws:

            async def send_audio():
                # Runs on a fixed cadence regardless of whether audio chunks are
                # arriving, so it still fires during real gaps (backgrounded tab,
                # suspended AudioContext, network stalls, a user-initiated pause)
                # — not just when a new chunk happens to show up. Munsit closes
                # the socket with 1011 after 12s of neither audio nor KeepAlive.
                stop_keepalive = asyncio.Event()

                async def keepalive_loop():
                    try:
                        while not stop_keepalive.is_set():
                            await asyncio.sleep(MUNSIT_KEEPALIVE_INTERVAL_S)
                            if stop_keepalive.is_set():
                                break
                            await ws.send(json.dumps({"type": "KeepAlive"}))
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    except Exception as e:
                        logger.warning("[Munsit] keepalive_loop error: %s", e)

                keepalive_task = asyncio.create_task(keepalive_loop())
                try:
                    async for chunk in audio_stream:
                        await ws.send(chunk)
                    # Audio source exhausted (recording stopped) — finalize the turn.
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    logger.warning("[Munsit] send_audio error: %s", e)
                finally:
                    stop_keepalive.set()
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        logger.debug("[Munsit] keepalive task ended: %s", exc)

            sender_task = asyncio.create_task(send_audio())

            try:
                async for raw in ws:
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "Results":
                        transcript = event.get("transcript", "") or ""
                        is_final = bool(event.get("is_final"))
                        if is_final:
                            if transcript:
                                yield {
                                    "type": "final",
                                    "text": transcript,
                                    "speech_final": bool(event.get("speech_final")),
                                }
                        else:
                            yield {"type": "interim", "text": transcript}

                    elif etype == "UtteranceEnd":
                        yield {"type": "utterance_end"}

                    elif etype == "Metadata":
                        if "audio_seconds_billed" in event:
                            # Closing Metadata — session is finishing.
                            yield {
                                "type": "done",
                                "billed_seconds": event.get("audio_seconds_billed"),
                            }
                            break
                        # else: opening Metadata — nothing actionable for the client.
                        elif event.get("dropped_hotwords"):
                            logger.warning(
                                "[Munsit] Hotwords dropped: %s", event["dropped_hotwords"]
                            )

                    elif etype == "Error":
                        recoverable = bool(event.get("recoverable"))
                        yield {
                            "type": "error",
                            "detail": event.get("message", "Munsit error"),
                            "code": event.get("code"),
                            "recoverable": recoverable,
                        }
                        if not recoverable:
                            break

                    # SpeechStarted / Gender / Sentiment: not surfaced to the
                    # dictation UI today; add handling here if you want them.

            finally:
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.debug("[Munsit] sender task ended: %s", exc)

    except websockets.exceptions.InvalidStatusCode as e:
        raise TranscriptionError(f"Munsit connection rejected: {e}") from e
    except Exception as e:
        raise TranscriptionError(f"Munsit stream failed: {e}") from e


# ---------------------------------------------------------------------------
# Munsit Text-to-Speech — resolve "Lina" (fusha) and synthesize the disclaimer
# ---------------------------------------------------------------------------

# voice_id is an opaque string per Munsit's docs (only a few are human-readable,
# and none are guaranteed enumerable) — it must be resolved by name via
# GET /voices rather than hardcoded or pattern-matched. Cached in-process since
# the voice catalogue changes rarely.
_voice_id_cache: Dict[str, str] = {}
_disclaimer_audio_cache: Optional[bytes] = None


async def _resolve_voice_id(api_key: str, voice_name: str, dialect: Optional[str] = None) -> str:
    cache_key = f"{voice_name.lower()}::{dialect or ''}"
    if cache_key in _voice_id_cache:
        return _voice_id_cache[cache_key]

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{MUNSIT_TTS_API_BASE}/voices", headers={"x-api-key": api_key})
        resp.raise_for_status()
        voices = resp.json()

    name_matches = [v for v in voices if (v.get("name") or "").strip().lower() == voice_name.lower()]
    if not name_matches:
        raise TranscriptionError(f"Munsit voice '{voice_name}' not found in voice library.")

    match = None
    if dialect:
        for v in name_matches:
            if dialect in (v.get("dialect") or []):
                match = v
                break
    if match is None:
        match = name_matches[0]  # fall back to the first name match

    voice_id = match["voice_id"]
    _voice_id_cache[cache_key] = voice_id
    logger.info("[MunsitTTS] Resolved voice '%s' (dialect=%s) -> %s", voice_name, dialect, voice_id)
    return voice_id


async def _synthesize_wav(
    api_key: str, text: str, voice_id: str, dialect: Optional[str] = None, speed: float = 1.0
) -> bytes:
    payload: Dict[str, Any] = {
        "voice_id": voice_id,
        "text": text,
        "stability": 0.5,
        "speed": speed,  # 0.7-1.2 per Munsit docs; 1.2 is the fastest allowed
        "streaming": False,
    }
    if dialect:
        payload["dialect"] = dialect

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{MUNSIT_TTS_API_BASE}/text-to-speech/{MUNSIT_TTS_MODEL}",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code != 200:
            raise TranscriptionError(f"Munsit TTS failed ({resp.status_code}): {resp.text}")
        return resp.content


async def get_recording_disclaimer_audio(api_key: str) -> bytes:
    """
    WAV bytes for the fixed "this session is being recorded" consent line,
    spoken by Lina in Fusha (MSA). The text is constant, so the rendered audio
    is cached in-process after the first successful synthesis.
    """
    global _disclaimer_audio_cache
    if _disclaimer_audio_cache is not None:
        return _disclaimer_audio_cache
    if not api_key:
        raise ValueError("Missing Munsit API key.")

    voice_id = await _resolve_voice_id(api_key, DISCLAIMER_VOICE_NAME, dialect=DISCLAIMER_DIALECT)
    # speed=1.2 is the fastest Munsit allows — shortens the clip itself. The
    # frontend additionally starts recording ~1s after playback begins rather
    # than waiting for the clip to finish, so total perceived delay stays low
    # regardless of clip length.
    audio_bytes = await _synthesize_wav(api_key, DISCLAIMER_TEXT, voice_id, dialect=DISCLAIMER_DIALECT, speed=1.2)
    _disclaimer_audio_cache = audio_bytes
    return audio_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SpeechService:
    """
    Speech recognition — routed by language:
      English → Mistral Voxtral real-time transcription
      Arabic  → Munsit real-time transcription (dialect-aware, incl. Khaliji/Gulf)

    Entry points:
      transcribe_audio()               – batch, from file  (Voxtral)
      transcribe_audio_stream()        – streaming deltas, from file  (Voxtral, post-recording)
      transcribe_live_stream()         – streaming deltas, from live WebSocket bytes  (Voxtral, English mic)
      transcribe_live_stream_ar()      – streaming events, from live WebSocket bytes  (Munsit, Arabic mic)
      get_recording_disclaimer_audio() – WAV bytes of the Munsit-TTS recording-consent line (Lina, Fusha)
    """

    # ------------------------------------------------------------------
    # Batch from file — Voxtral
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
    # Streaming deltas from file — Voxtral post-recording fallback
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
    # Streaming deltas from live WebSocket bytes — Voxtral, English mic
    # ------------------------------------------------------------------
    @staticmethod
    async def transcribe_live_stream(
        raw_pcm_iter: AsyncIterator[bytes],
        api_key: str,
        input_sample_rate: int = 16000,
    ) -> AsyncIterator[str]:
        """
        Accept a live async iterator of float32 PCM bytes (from browser AudioContext),
        convert via ffmpeg (resampling from input_sample_rate → 16kHz), stream to
        Voxtral, yield text deltas in real time.

        Note: unlike the Munsit path, this stream has no application-level
        KeepAlive — a client-side pause simply means no new chunks arrive from
        raw_pcm_iter for a while. This has not been an issue in testing, but if
        Mistral's realtime session ever enforces its own server-side idle
        timeout, a long pause in English mode could close the socket; watch for
        that if pauses regularly exceed ~30-60s in English sessions.
        """
        audio_stream = _ffmpeg_stdin_stream(
            raw_pcm_iter,
            input_sample_rate=input_sample_rate,
        )
        async for delta in _transcribe_yield(audio_stream, api_key):
            yield delta

    # ------------------------------------------------------------------
    # Streaming events from live WebSocket bytes — Munsit, Arabic mic
    # ------------------------------------------------------------------
    @staticmethod
    async def transcribe_live_stream_ar(
        raw_pcm_iter: AsyncIterator[bytes],
        api_key: str,
        input_sample_rate: int = 16000,
        hotwords: Optional[str] = None,
        code_switching: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Accept a live async iterator of float32 PCM bytes (from browser AudioContext),
        convert via ffmpeg to 16kHz linear16, stream to Munsit, yield normalized
        transcription events (see _transcribe_munsit_yield for the event shapes).

        Args:
            hotwords:       Optional comma-separated custom vocabulary
                             (e.g. medical terms, drug names) — up to 200 entries.
            code_switching: Use the munsit-en-ar model for mixed Arabic/English
                             speech instead of the pure-Arabic model.
        """
        audio_stream = _ffmpeg_stdin_stream(
            raw_pcm_iter,
            input_sample_rate=input_sample_rate,
        )
        model = "munsit-en-ar" if code_switching else MUNSIT_MODEL
        async for event in _transcribe_munsit_yield(audio_stream, api_key, hotwords=hotwords, model=model):
            yield event

    # ------------------------------------------------------------------
    # Munsit TTS — recording-consent disclaimer (Lina, Fusha)
    # ------------------------------------------------------------------
    @staticmethod
    async def get_recording_disclaimer_audio(api_key: str) -> bytes:
        """WAV bytes of the recording-consent line, played once at session start."""
        return await get_recording_disclaimer_audio(api_key)

    @staticmethod
    def transcribe_audio_sync(audio_file_path: str, api_key: str, **kwargs):
        return asyncio.run(SpeechService.transcribe_audio(audio_file_path, api_key, **kwargs))