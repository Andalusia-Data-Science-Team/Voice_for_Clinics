"""Locust load test for MedVoice live transcription.

The application exposes one WebSocket endpoint and routes by the initial
language message:

* ``en`` -> Mistral Voxtral realtime transcription;
* ``ar`` -> Munsit realtime transcription.

This test sends generated, non-speech PCM audio by default. It exercises the
WebSocket lifecycle, PCM conversion, provider session handling, and result
streaming without using real recordings or PHI. For a speech-quality test,
point ``LOCUST_PCM_FILE`` at an approved synthetic float32 PCM file.

Examples::

    locust -f locustfile_transcription.py --host http://localhost:7777
    locust -f locustfile_transcription.py --host http://localhost:7777 \
        --headless -u 10 -r 2 -t 2m

Useful environment variables::

    LOCUST_TRANSCRIPTION_LANGUAGE=en|ar  Default: en
    LOCUST_TRANSCRIPTION_LANGUAGE_MIX=en:50,ar:50
                                           Used when the single language is unset
    LOCUST_TRANSCRIPTION_SECONDS=5         Synthetic audio duration
    LOCUST_TRANSCRIPTION_SAMPLE_RATE=16000
    LOCUST_TRANSCRIPTION_CHUNK_MS=100      PCM chunk duration
    LOCUST_TRANSCRIPTION_REALTIME=true     Pace chunks in real time
    LOCUST_TRANSCRIPTION_TIMEOUT=60        WebSocket/result timeout
    LOCUST_PCM_FILE=                       Optional raw float32 PCM file
    LOCUST_REQUIRE_RESULT=true             Fail when no transcription text
                                           arrives; set to false to treat a
                                           completed session over non-speech
                                           audio as a successful lifecycle run

The server owns the Mistral and Munsit credentials. Use a staging server and
do not point this test at real patient recordings.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import websocket
from locust import User, between, events, task
from locust.env import Environment


def _parse_mix(raw: str) -> tuple[list[str], list[int]]:
    """Parse a language mix such as ``en:50,ar:50``."""
    values: dict[str, int] = {}
    for item in raw.split(","):
        language, weight = item.strip().split(":", 1)
        if language not in {"en", "ar"} or int(weight) <= 0:
            raise ValueError(f"Invalid transcription language mix item: {item!r}")
        values[language] = int(weight)
    return list(values), list(values.values())


def _synthetic_pcm(seconds: float, sample_rate: int) -> bytes:
    """Create low-amplitude synthetic PCM that contains no speech or PHI."""
    samples = int(seconds * sample_rate)
    # A quiet tone is more useful than all-zero input for exercising the
    # provider's audio path, while remaining clearly non-speech test data.
    return b"".join(
        struct.pack("<f", 0.02 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(samples)
    )


def _fire_metric(
    environment: Environment,
    name: str,
    response_time: float,
    response_length: int = 0,
    exception: Exception | None = None,
) -> None:
    """Record a transcription timing row in Locust statistics."""
    environment.events.request.fire(
        request_type="WS",
        name=name,
        response_time=response_time,
        response_length=response_length,
        exception=exception,
        context={},
    )


def _failure_name(language: str, error: BaseException) -> str:
    """Create a stable failure name grouped by provider and failure type."""
    text = str(error).lower()
    if "timeout" in text:
        kind = "timeout"
    elif "connection" in text or "closed" in text:
        kind = "connection error"
    elif "error" in text:
        kind = "provider error"
    else:
        kind = "transcription error"
    provider = "Voxtral" if language == "en" else "Munsit"
    return f"{provider} {kind}"


class LiveTranscriptionUser(User):
    """Open a live transcription session and consume it to completion."""

    wait_time = between(1, 3)
    ws_path = os.getenv("LOCUST_TRANSCRIPTION_PATH", "/ws/transcribe")
    timeout = float(os.getenv("LOCUST_TRANSCRIPTION_TIMEOUT", "60"))
    sample_rate = int(os.getenv("LOCUST_TRANSCRIPTION_SAMPLE_RATE", "16000"))
    chunk_ms = int(os.getenv("LOCUST_TRANSCRIPTION_CHUNK_MS", "100"))
    audio_seconds = float(os.getenv("LOCUST_TRANSCRIPTION_SECONDS", "5"))
    realtime = os.getenv("LOCUST_TRANSCRIPTION_REALTIME", "true").lower() in {
        "1", "true", "yes",
    }
    pcm_file = os.getenv("LOCUST_PCM_FILE", "").strip()
    require_result = os.getenv("LOCUST_REQUIRE_RESULT", "true").lower() in {
        "1", "true", "yes",
    }
    language_names, language_weights = _parse_mix(
        os.getenv("LOCUST_TRANSCRIPTION_LANGUAGE_MIX", "en:50,ar:50")
    )

    def _audio(self) -> bytes:
        """Load approved raw float32 PCM or generate synthetic test audio."""
        if self.pcm_file:
            return Path(self.pcm_file).read_bytes()
        return _synthetic_pcm(self.audio_seconds, self.sample_rate)

    @task
    def transcribe(self) -> None:
        """Run one Voxtral or Munsit WebSocket transcription session."""
        language_override = os.getenv("LOCUST_TRANSCRIPTION_LANGUAGE", "").strip()
        if language_override and language_override not in {"en", "ar"}:
            raise ValueError("LOCUST_TRANSCRIPTION_LANGUAGE must be en or ar")
        language = language_override or __import__("random").choices(
            self.language_names, weights=self.language_weights, k=1
        )[0]
        started = time.perf_counter()
        received_bytes = 0
        result_count = 0
        first_result_at: float | None = None
        done = Event()
        receiver_error: list[BaseException] = []
        ws = None

        try:
            base_url = self.host or "http://localhost:7777"
            ws_url = base_url.replace("https://", "wss://").replace(
                "http://", "ws://"
            ) + self.ws_path
            ws = websocket.create_connection(ws_url, timeout=self.timeout)

            def receive_results() -> None:
                nonlocal received_bytes, result_count, first_result_at
                try:
                    while not done.is_set():
                        raw = ws.recv()
                        if raw is None:
                            break
                        if isinstance(raw, str):
                            received_bytes += len(raw.encode("utf-8"))
                            try:
                                event: dict[str, Any] = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            event_type = event.get("type")
                            if event_type in {"delta", "interim"}:
                                result_count += 1
                                if first_result_at is None:
                                    first_result_at = (
                                        time.perf_counter() - started
                                    ) * 1000
                            if event_type == "error":
                                detail = str(event.get("detail", "provider error"))
                                receiver_error.append(RuntimeError(detail))
                                break
                            if event_type == "done":
                                break
                        else:
                            received_bytes += len(raw)
                except Exception as exc:
                    receiver_error.append(exc)

            receiver = Thread(target=receive_results, daemon=True)
            receiver.start()
            ws.send(json.dumps({"sampleRate": self.sample_rate, "language": language}))
            pcm = self._audio()
            bytes_per_chunk = max(8, self.sample_rate * 4 * self.chunk_ms // 1000)
            chunk_delay = self.chunk_ms / 1000
            for offset in range(0, len(pcm), bytes_per_chunk):
                chunk = pcm[offset:offset + bytes_per_chunk]
                ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
                if self.realtime:
                    time.sleep(chunk_delay)
            ws.send(json.dumps({"cmd": "done"}))
            receiver.join(timeout=self.timeout)
            if receiver.is_alive():
                raise TimeoutError("transcription result timeout")
            if receiver_error:
                raise receiver_error[0]
            if result_count == 0 and self.require_result:
                raise RuntimeError("transcription completed without a result event")
            elapsed_ms = (time.perf_counter() - started) * 1000
            _fire_metric(
                self.environment,
                f"WS /ws/transcribe ({'Voxtral' if language == 'en' else 'Munsit'})",
                elapsed_ms,
                received_bytes,
            )
            if first_result_at is not None:
                _fire_metric(
                    self.environment,
                    f"WS first result ({'Voxtral' if language == 'en' else 'Munsit'})",
                    first_result_at,
                    received_bytes,
                )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            _fire_metric(
                self.environment,
                f"WS /ws/transcribe ({'Voxtral' if language == 'en' else 'Munsit'})",
                elapsed_ms,
                received_bytes,
                exc,
            )
        finally:
            done.set()
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


@events.quitting.add_listener
def _print_summary(environment: Environment) -> None:
    """Print a compact transcription load-test summary."""
    for provider in ("Voxtral", "Munsit"):
        entry = environment.stats.get(f"WS /ws/transcribe ({provider})", "WS")
        p95 = entry.get_response_time_percentile(0.95) if entry.num_requests else 0
        print(
            f"{provider}: requests={entry.num_requests}, "
            f"failures={entry.num_failures}, p95={p95} ms"
        )