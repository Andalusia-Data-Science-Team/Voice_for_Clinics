"""Locust stress test for the MedVoice HTTP analysis pipeline.

The main task targets ``POST /audio/analyze`` because it exercises the shared
LLM runtime, per-model limiters, retries, and SSE streaming response. The
request uses synthetic clinical text and does not upload audio or send PHI.
The main Locust row measures the complete SSE stream, not only response headers.

Examples::

    locust -f locustfile.py --host http://localhost:7777
    locust -f locustfile.py --host http://localhost:7777 \
        --headless -u 30 -r 5 -t 5m
    locust -f locustfile.py --host http://localhost:7777 \
        --headless -u 30 -r 8 -t 5m --csv results --html report.html

Useful environment variables::

    LOCUST_LANGUAGE=en|ar             Default: en
    LOCUST_MODE=doctor|conversation   Default: doctor
    LOCUST_LANGUAGE_MIX=en:70,ar:30   Used when LOCUST_LANGUAGE is unset
    LOCUST_MODE_MIX=doctor:70,conversation:30
                                      Used when LOCUST_MODE is unset
    LOCUST_DEPARTMENT=er|opd|dental   Default: opd
    LOCUST_SPECIALTY=                 Optional specialty value
    LOCUST_ANALYZE_TIMEOUT=300        Streaming request timeout in seconds
    LOCUST_WAIT_MIN=1                 Minimum wait between tasks per user
    LOCUST_WAIT_MAX=3                 Maximum wait between tasks per user
    LOCUST_LENGTH_MIX=short:50,medium:35,long:15
    LOCUST_MAX_FAIL_RATIO=0.01        Headless exit threshold; default 1%
    LOCUST_MAX_P95_MS=                Optional final-extraction p95 threshold
    LOCUST_WS_MAX_FAIL_RATIO=         ws-session row fail ratio (defaults to
                                      LOCUST_MAX_FAIL_RATIO)
    LOCUST_WS_MAX_P95_MS=             Optional WS-session total p95 threshold
    LOCUST_SESSION_CSV=locust_sessions.csv
                                      Per-session CSV path
    LOCUST_SESSION_CSV_APPEND=false   Append instead of truncating per run

The optional stepped shape is in ``step_load_shape.py``. Use both files to
enable it, for example ``locust -f locustfile.py,step_load_shape.py``.
Its defaults are 10, 20, 40, 60, 80, and 100 users for two minutes each.
Configure it with ``LOCUST_SHAPE_STEPS``, ``LOCUST_SHAPE_STEP_SECONDS``, and
``LOCUST_SHAPE_SPAWN_RATE``.

The application must be configured with valid provider credentials before
running this test. Run against a staging environment whenever possible.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import struct
import threading
import time
import wave
from typing import Any
from uuid import uuid4

import requests
from locust import HttpUser, between, events, task
from locust.env import Environment


SYNTHETIC_TRANSCRIPTS: dict[str, dict[str, str]] = {
    "en": {
        "short": (
            "Synthetic load-test visit. Mild headache for two days. "
            "Vital signs are stable."
        ),
        "medium": (
            "Synthetic load-test visit for performance testing only. The fictional "
            "patient reports intermittent headache and mild fatigue for four days. "
            "No red-flag symptoms are reported. Examination is unremarkable and "
            "vital signs are stable. The clinician recommends routine follow-up."
        ),
        "long": (
            "Synthetic load-test visit for performance testing only; this is not a "
            "real patient record. The fictional patient reports intermittent "
            "headache, mild fatigue, and occasional difficulty concentrating for "
            "approximately one week. The symptoms are described as mild and are not "
            "associated with loss of consciousness, weakness, fever, chest pain, or "
            "shortness of breath. The fictional patient reports adequate hydration "
            "and no recent medication change. A synthetic examination notes stable "
            "vital signs, normal orientation, and no focal neurological finding. "
            "The clinician discusses observation, routine supportive care, and "
            "follow-up if symptoms persist or become more severe. All details in "
            "this paragraph are generated test data for load testing."
        ),
    },
    "ar": {
        "short": (
            "زيارة اختبارية اصطناعية. صداع خفيف لمدة يومين. العلامات الحيوية مستقرة."
        ),
        "medium": (
            "هذه زيارة اصطناعية لاختبار الأداء فقط. يذكر المريض الافتراضي صداعاً "
            "متقطعاً وإرهاقاً خفيفاً منذ أربعة أيام، دون أعراض خطيرة. الفحص مستقر "
            "والعلامات الحيوية طبيعية، ويوصى بمتابعة روتينية."
        ),
        "long": (
            "هذه زيارة اصطناعية لاختبار الأداء وليست سجلاً لمريض حقيقي. يذكر المريض "
            "الافتراضي صداعاً متقطعاً وإرهاقاً خفيفاً وصعوبة بسيطة في التركيز منذ "
            "نحو أسبوع. الأعراض خفيفة ولا يصاحبها فقدان وعي أو ضعف أو حمى أو ألم "
            "في الصدر أو ضيق في التنفس. يذكر المريض الافتراضي شرب كمية كافية من "
            "الماء وعدم تغيير الأدوية مؤخراً. يوضح الفحص الاصطناعي استقرار العلامات "
            "الحيوية وسلامة الوعي وعدم وجود علامة عصبية بؤرية. يناقش الطبيب "
            "المتابعة والعلاج الداعم ومراجعة العيادة إذا استمرت الأعراض أو ازدادت. "
            "جميع التفاصيل هنا بيانات اختبار مولدة اصطناعياً لاختبار الضغط فقط."
        ),
    },
}

SYNTHETIC_CONVERSATION_TRANSCRIPTS: dict[str, dict[str, str]] = {
    "en": {
        "short": (
            "Doctor: What brings you in today? Patient: I have a mild headache "
            "that started two days ago. Doctor: Any warning symptoms? Patient: No, "
            "and I feel otherwise well."
        ),
        "medium": (
            "Doctor: Please describe the problem. Patient: I have had intermittent "
            "headache and mild fatigue for four days. Doctor: Any fever, weakness, "
            "chest pain, or shortness of breath? Patient: No. Doctor: Have you "
            "changed any medicines recently? Patient: No, and I am drinking water "
            "normally. Doctor: We will document this synthetic visit and arrange "
            "routine follow-up."
        ),
        "long": (
            "Doctor: Tell me what has been happening. Patient: I have experienced "
            "intermittent headache, mild fatigue, and occasional difficulty "
            "concentrating for about one week. Doctor: Did you lose consciousness, "
            "develop weakness, fever, chest pain, or shortness of breath? Patient: "
            "No, none of those symptoms. Doctor: Any recent medication change or "
            "reduced fluid intake? Patient: No medication change, and I am drinking "
            "adequately. Doctor: The synthetic examination shows stable vital signs, "
            "normal orientation, and no focal neurological finding. We will discuss "
            "supportive care and follow-up if the symptoms persist or worsen. This "
            "conversation contains generated test data only."
        ),
    },
    "ar": {
        "short": (
            "الطبيب: ما سبب الزيارة؟ المريض: لدي صداع خفيف بدأ منذ يومين. "
            "الطبيب: هل توجد أعراض خطيرة؟ المريض: لا، وأشعر أن حالتي مستقرة."
        ),
        "medium": (
            "الطبيب: صف لي المشكلة. المريض: أعاني من صداع متقطع وإرهاق خفيف منذ "
            "أربعة أيام. الطبيب: هل توجد حمى أو ضعف أو ألم في الصدر أو ضيق تنفس؟ "
            "المريض: لا. الطبيب: هل غيرت أي دواء مؤخراً؟ المريض: لا، وأشرب الماء "
            "بشكل طبيعي. الطبيب: سنوثق هذه الزيارة الاصطناعية ونحدد متابعة روتينية."
        ),
        "long": (
            "الطبيب: أخبرني بما حدث. المريض: أعاني من صداع متقطع وإرهاق خفيف "
            "وصعوبة بسيطة في التركيز منذ نحو أسبوع. الطبيب: هل حدث فقدان للوعي أو "
            "ضعف أو حمى أو ألم في الصدر أو ضيق في التنفس؟ المريض: لا، لم يحدث أي "
            "من ذلك. الطبيب: هل غيرت الأدوية مؤخراً أو قل شربك للسوائل؟ المريض: "
            "لا يوجد تغيير في الأدوية وأشرب كمية كافية. الطبيب: الفحص الاصطناعي "
            "يظهر استقرار العلامات الحيوية وسلامة الوعي وعدم وجود علامة عصبية بؤرية. "
            "سنناقش العلاج الداعم والمتابعة إذا استمرت الأعراض أو ازدادت. هذه "
            "المحادثة تحتوي على بيانات اختبار مولدة اصطناعياً فقط."
        ),
    },
}

SSE_STAGE_NAMES = {
    "refinement_skipped",
    "refinement_delta",
    "refinement_done",
    "translation_delta",
    "translation_done",
    "error",
    "extraction",
}


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    """Read and validate a simple environment-backed option."""
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


def _parse_length_mix(raw: str) -> tuple[list[str], list[int]]:
    """Parse ``short:50,medium:35,long:15`` into choices and weights."""
    weights: dict[str, int] = {}
    for item in raw.split(","):
        try:
            length, weight_text = item.strip().split(":", 1)
            weight = int(weight_text)
        except ValueError as exc:
            raise ValueError(f"Invalid LOCUST_LENGTH_MIX item: {item!r}") from exc
        if length not in {"short", "medium", "long"} or weight <= 0:
            raise ValueError(f"Invalid LOCUST_LENGTH_MIX item: {item!r}")
        weights[length] = weight
    if not weights:
        raise ValueError("LOCUST_LENGTH_MIX must contain at least one length")
    return list(weights), list(weights.values())


def _parse_choice_mix(
    raw: str, choices: set[str], setting_name: str
) -> tuple[list[str], list[int]]:
    """Parse a weighted mix such as ``en:70,ar:30``."""
    weights: dict[str, int] = {}
    for item in raw.split(","):
        try:
            choice, weight_text = item.strip().split(":", 1)
            weight = int(weight_text)
        except ValueError as exc:
            raise ValueError(f"Invalid {setting_name} item: {item!r}") from exc
        if choice not in choices or weight <= 0:
            raise ValueError(f"Invalid {setting_name} item: {item!r}")
        weights[choice] = weight
    if not weights:
        raise ValueError(f"{setting_name} must contain at least one choice")
    return list(weights), list(weights.values())


def _fire_sse_metric(environment: Environment, name: str, response_time: float) -> None:
    """Record one custom SSE timing as a separate Locust statistics row."""
    environment.events.request.fire(
        request_type="SSE",
        name=name,
        response_time=response_time,
        response_length=0,
        exception=None,
        context={},
    )


def _failure_category(error: BaseException) -> str:
    """Return a stable failure category for Locust's Failures tab."""
    if isinstance(error, requests.exceptions.Timeout):
        return "Timeout"
    if isinstance(error, requests.exceptions.ConnectionError):
        return "Connection error"
    # The WebSocket phase raises websocket-client exceptions, which are not
    # ``requests`` exceptions; classify them explicitly so they do not all
    # collapse into the generic "Stream error" bucket.
    try:
        import websocket
    except ImportError:  # websocket-client is optional for HTTP-only runs
        websocket = None
    if websocket is not None:
        if isinstance(error, websocket.WebSocketTimeoutException):
            return "Timeout"
        if isinstance(error, websocket.WebSocketException):
            return "WS error"
    return "Stream error"


def _percentile(entry: Any, percentile: float) -> int:
    """Return a percentile, or zero when a metric has no samples."""
    return entry.get_response_time_percentile(percentile) if entry.num_requests else 0


# ── Per-session CSV (A5.2.1) ───────────────────────────────────────────────
# One row per session: first partial, final transcript, first SSE event, total
# time. Written to LOCUST_SESSION_CSV (default: locust_sessions.csv).
_SESSION_CSV_PATH = os.getenv("LOCUST_SESSION_CSV", "locust_sessions.csv")
_session_csv_lock = threading.Lock()
_session_csv_header_written = False
_session_csv_run_tag = ""
_session_csv_target_path = _SESSION_CSV_PATH


def _session_csv_tag(environment: Environment) -> str:
    """Return a stable per-run, per-process tag for the session CSV."""
    explicit = os.getenv("LOCUST_SESSION_CSV_TAG", "").strip()
    if explicit:
        return explicit
    runner = getattr(environment, "runner", None)
    role = "local"
    if runner is not None:
        name = type(runner).__name__.lower()
        if "master" in name:
            role = "master"
        elif "worker" in name:
            role = "worker"
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{role}-{stamp}-{os.getpid()}"


def _session_csv_target(environment: Environment) -> str:
    """Resolve the CSV path for this process, once per run.

    A distributed run (``--master``/``--worker``) gives every process its own
    file so workers never interleave rows or race on the header. A local run
    keeps the configured path and is truncated at ``test_start`` so rows do not
    accumulate across runs.
    """
    global _session_csv_run_tag, _session_csv_target_path
    if _session_csv_run_tag:
        return _session_csv_target_path
    _session_csv_run_tag = _session_csv_tag(environment)
    runner = getattr(environment, "runner", None)
    name = type(runner).__name__.lower() if runner is not None else "local"
    if "master" in name or "worker" in name:
        root, ext = os.path.splitext(_SESSION_CSV_PATH)
        _session_csv_target_path = f"{root}.{_session_csv_run_tag}{ext or '.csv'}"
    else:
        _session_csv_target_path = _SESSION_CSV_PATH
    return _session_csv_target_path


@events.test_start.add_listener
def _reset_session_csv(environment: Environment, **kwargs: Any) -> None:
    """Start each run with an empty CSV so rows never mix across runs."""
    global _session_csv_header_written
    target = _session_csv_target(environment)
    append = os.getenv("LOCUST_SESSION_CSV_APPEND", "").lower() in {
        "1", "true", "yes",
    }
    with _session_csv_lock:
        _session_csv_header_written = False
        if append:
            return
        try:
            with open(target, "w", encoding="utf-8"):
                pass
        except OSError:
            pass


def _write_session_row(environment: Environment, row: dict[str, Any]) -> None:
    """Append one per-session row, writing the header once per run."""
    global _session_csv_header_written
    fields = [
        "session_id", "language", "mode",
        "first_partial", "final_transcript", "first_sse_event", "total_sec",
    ]
    target = _session_csv_target(environment)
    with _session_csv_lock:
        write_header = not _session_csv_header_written
        with open(target, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()
                _session_csv_header_written = True
            writer.writerow({k: row.get(k, "") for k in fields})


# ── Synthetic WAV (A5.1.1) ─────────────────────────────────────────────────
def _synthetic_wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    """Return a synthetic 16-bit mono WAV tone (no speech, no PHI)."""
    frames = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        samples = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(frames)
        )
        wav.writeframes(samples)
    return buf.getvalue()


def _wav_to_pcm_chunks(wav_bytes: bytes, chunk_ms: int = 100) -> tuple[list[bytes], int]:
    """Decode a WAV into float32 little-endian PCM chunks and its sample rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    # Convert to mono float32.
    if sampwidth == 2:
        import array
        ints = array.array("h")
        ints.frombytes(raw)
        if n_channels > 1:
            ints = array.array("h", ints[0::n_channels])
        floats = [s / 32768.0 for s in ints]
    else:
        floats = [0.0] * (len(raw) // max(1, sampwidth))

    samples_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    chunks: list[bytes] = []
    for start in range(0, len(floats), samples_per_chunk):
        piece = floats[start:start + samples_per_chunk]
        chunks.append(struct.pack(f"<{len(piece)}f", *piece))
    return chunks, sample_rate


def _fail_ratio_reason(
    entry: Any, threshold: float, description: str
) -> str | None:
    """Return a threshold-failure reason for a row's failure ratio, if any."""
    if not entry.num_requests:
        return None
    ratio = entry.num_failures / entry.num_requests
    if ratio > threshold:
        return f"{description} failure ratio {ratio:.2%} > {threshold:.2%}"
    return None


def _p95_reason(entry: Any, max_ms: float, description: str) -> str | None:
    """Return a threshold-failure reason for a row's p95 latency, if any."""
    p95 = _percentile(entry, 0.95)
    if not entry.num_requests or p95 > max_ms:
        return f"{description} p95 {p95} ms > {max_ms:g} ms"
    return None


@events.quitting.add_listener
def _check_thresholds(environment: Environment) -> None:
    """Print end-of-test metrics and set a failing headless exit code."""
    stats = environment.stats
    main = stats.get("POST /audio/analyze", "POST")
    first = stats.get("SSE first event", "SSE")
    final = stats.get("SSE final extraction", "SSE")
    ws_main = stats.get("POST /audio/analyze (ws-session)", "POST")
    ws_total = stats.get("WS session total", "SSE")
    failure_ratio = main.num_failures / main.num_requests if main.num_requests else 0.0

    max_fail_ratio = float(os.getenv("LOCUST_MAX_FAIL_RATIO", "0.01"))
    ws_max_fail_ratio = float(
        os.getenv("LOCUST_WS_MAX_FAIL_RATIO", str(max_fail_ratio))
    )
    max_p95_text = os.getenv("LOCUST_MAX_P95_MS", "").strip()
    ws_max_p95_text = os.getenv("LOCUST_WS_MAX_P95_MS", "").strip()

    print(
        "\nLocust summary: "
        f"requests={main.num_requests}, failures={main.num_failures}, "
        f"failure_ratio={failure_ratio:.2%}; "
        f"ws-sessions={ws_main.num_requests}, ws-failures={ws_main.num_failures}"
    )
    print(
        "Full stream ms: "
        f"p50={_percentile(main, 0.50)}, p95={_percentile(main, 0.95)}, "
        f"p99={_percentile(main, 0.99)}"
    )
    print(
        "First event ms: "
        f"p50={_percentile(first, 0.50)}, p95={_percentile(first, 0.95)}, "
        f"p99={_percentile(first, 0.99)}"
    )
    print(
        "WS session ms: "
        f"p50={_percentile(ws_total, 0.50)}, p95={_percentile(ws_total, 0.95)}, "
        f"p99={_percentile(ws_total, 0.99)}"
    )

    reasons: list[str] = []
    for reason in (
        _fail_ratio_reason(main, max_fail_ratio, "POST /audio/analyze"),
        _fail_ratio_reason(
            ws_main, ws_max_fail_ratio, "POST /audio/analyze (ws-session)"
        ),
    ):
        if reason:
            reasons.append(reason)
    if max_p95_text:
        reason = _p95_reason(final, float(max_p95_text), "final extraction")
        if reason:
            reasons.append(reason)
    if ws_max_p95_text:
        reason = _p95_reason(ws_total, float(ws_max_p95_text), "WS session total")
        if reason:
            reasons.append(reason)
    if reasons:
        environment.process_exit_code = 1
        print("Threshold failure: " + "; ".join(reasons))


class MedVoiceAnalyzeUser(HttpUser):
    """Generate realistic concurrent SSE analysis requests."""

    # Each user waits between completed analysis requests. The number of
    # simultaneous users is controlled by Locust's -u/--users option.
    wait_time = between(
        float(os.getenv("LOCUST_WAIT_MIN", "1")),
        float(os.getenv("LOCUST_WAIT_MAX", "3")),
    )

    analyze_path = os.getenv("LOCUST_ANALYZE_PATH", "/audio/analyze")
    analyze_timeout = float(os.getenv("LOCUST_ANALYZE_TIMEOUT", "300"))
    length_names, length_weights = _parse_length_mix(
        os.getenv("LOCUST_LENGTH_MIX", "short:50,medium:35,long:15")
    )
    language_names, language_weights = _parse_choice_mix(
        os.getenv("LOCUST_LANGUAGE_MIX", "en:70,ar:30"),
        {"en", "ar"},
        "LOCUST_LANGUAGE_MIX",
    )
    mode_names, mode_weights = _parse_choice_mix(
        os.getenv("LOCUST_MODE_MIX", "doctor:70,conversation:30"),
        {"doctor", "conversation"},
        "LOCUST_MODE_MIX",
    )

    def on_start(self) -> None:
        # The legacy single-value settings remain supported and take precedence
        # over the new per-request weighted mixes when explicitly provided.
        self.language_override = (
            _env_choice("LOCUST_LANGUAGE", "", {"en", "ar"})
            if os.getenv("LOCUST_LANGUAGE")
            else None
        )
        self.mode_override = (
            _env_choice("LOCUST_MODE", "", {"doctor", "conversation"})
            if os.getenv("LOCUST_MODE")
            else None
        )
        self.department = _env_choice(
            "LOCUST_DEPARTMENT", "opd", {"er", "opd", "dental"}
        )
        self.specialty = os.getenv("LOCUST_SPECIALTY", "").strip()

    @task
    def analyze_transcript(self) -> None:
        """Submit one synthetic transcript and consume the complete SSE stream."""
        language = self.language_override or random.choices(
            self.language_names, weights=self.language_weights, k=1
        )[0]
        mode = self.mode_override or random.choices(
            self.mode_names, weights=self.mode_weights, k=1
        )[0]
        length = random.choices(self.length_names, weights=self.length_weights, k=1)[0]
        data: dict[str, Any] = {
            "text": (
                SYNTHETIC_CONVERSATION_TRANSCRIPTS[language][length]
                if mode == "conversation"
                else SYNTHETIC_TRANSCRIPTS[language][length]
            ),
            "language": language,
            "mode": mode,
            "department": self.department,
            "use_history": "false",
            "session_id": f"locust-{uuid4().hex}",
            "clinician_name": "locust-load-test",
        }
        if self.specialty:
            data["specialty"] = self.specialty

        started = time.perf_counter()
        with self.client.post(
            self.analyze_path,
            data=data,
            name="POST /audio/analyze",
            stream=True,
            timeout=self.analyze_timeout,
            catch_response=True,
        ) as response:
            events_seen: set[str] = set()
            current_event = "message"
            stream_error = False
            first_event_recorded = False
            total_bytes = 0
            failure_reason: str | None = None

            try:
                if response.error is not None:
                    failure_reason = _failure_category(response.error)
                elif response.status_code != 200:
                    # Reading the body makes response_length meaningful for an
                    # immediate HTTP failure as well as for a successful stream.
                    body = response.content or b""
                    total_bytes = len(body)
                    failure_reason = f"HTTP {response.status_code}"
                else:
                    for raw_line in response.iter_lines(decode_unicode=False):
                        line_bytes = raw_line or b""
                        if isinstance(line_bytes, str):
                            line_bytes = line_bytes.encode("utf-8")
                        total_bytes += len(line_bytes) + 1
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        if not first_event_recorded:
                            _fire_sse_metric(
                                self.environment, "SSE first event", elapsed_ms
                            )
                            first_event_recorded = True

                        if line.startswith("event:"):
                            current_event = line.removeprefix("event:").strip()
                            if (
                                current_event in SSE_STAGE_NAMES
                                and current_event not in events_seen
                            ):
                                events_seen.add(current_event)
                                metric_name = (
                                    "SSE final extraction"
                                    if current_event == "extraction"
                                    else f"SSE {current_event}"
                                )
                                _fire_sse_metric(
                                    self.environment, metric_name, elapsed_ms
                                )
                        elif line.startswith("data:") and current_event == "error":
                            stream_error = True

                    if stream_error:
                        failure_reason = "SSE error event"
                    elif "extraction" not in events_seen:
                        failure_reason = "SSE stream ended without extraction"
            except requests.exceptions.Timeout as exc:
                failure_reason = _failure_category(exc)
            except requests.exceptions.ConnectionError as exc:
                failure_reason = _failure_category(exc)
            except requests.exceptions.RequestException:
                failure_reason = "Stream error"
            except Exception:
                # Do not let an unexpected client-side stream error kill a
                # Locust user; classify it as a failed request instead.
                failure_reason = "Stream error"
            finally:
                response.request_meta["response_time"] = (
                    time.perf_counter() - started
                ) * 1000
                response.request_meta["response_length"] = total_bytes
                if failure_reason:
                    response.failure(failure_reason)
                else:
                    response.success()

    @task
    def stream_wav_and_analyze(self) -> None:
        """A5.1.1/A5.2.1: stream a synthetic WAV over the WebSocket in real
        time, then read SSE until "done", and write one per-session CSV row.

        The WAV is a synthetic tone (or an approved file via LOCUST_WAV_FILE);
        never PHI. Runs for both languages.
        """
        language = self.language_override or random.choices(
            self.language_names, weights=self.language_weights, k=1
        )[0]
        mode = self.mode_override or random.choices(
            self.mode_names, weights=self.mode_weights, k=1
        )[0]
        session_id = f"locust-{uuid4().hex}"

        wav_path = os.getenv("LOCUST_WAV_FILE", "").strip()
        if wav_path and os.path.exists(wav_path):
            with open(wav_path, "rb") as fh:
                wav_bytes = fh.read()
        else:
            wav_bytes = _synthetic_wav_bytes(
                float(os.getenv("LOCUST_WAV_SECONDS", "3")),
                int(os.getenv("LOCUST_WAV_SAMPLE_RATE", "16000")),
            )
        chunks, sample_rate = _wav_to_pcm_chunks(
            wav_bytes, int(os.getenv("LOCUST_WAV_CHUNK_MS", "100"))
        )

        ws_url = self.host.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.rstrip("/") + "/ws/transcribe"

        first_partial = ""
        final_transcript = ""
        first_sse_event = ""
        started = time.perf_counter()
        failure_reason: str | None = None

        try:
            import websocket  # websocket-client

            ws = websocket.create_connection(
                ws_url, timeout=float(os.getenv("LOCUST_WS_TIMEOUT", "60"))
            )
            try:
                ws.send(json.dumps({"sampleRate": sample_rate, "language": language}))
                chunk_interval = float(os.getenv("LOCUST_WAV_CHUNK_MS", "100")) / 1000.0
                for chunk in chunks:
                    ws.send_binary(chunk)
                    time.sleep(chunk_interval)  # real-time pacing
                ws.send(json.dumps({"cmd": "done"}))

                while True:
                    message = ws.recv()
                    if not message:
                        break
                    try:
                        payload = json.loads(message)
                    except Exception:
                        continue
                    mtype = payload.get("type")
                    if mtype == "interim" and not first_partial:
                        first_partial = payload.get("text", "")
                    elif mtype == "delta":
                        final_transcript += payload.get("text", "")
                    elif mtype == "done":
                        break
                    elif mtype == "error":
                        failure_reason = "WS error event"
                        break
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
        except Exception as exc:
            failure_reason = _failure_category(exc)

        # The analyze SSE contract has no generic "done" event; consume until
        # the response closes so diagnosis/validation are included in the run.
        saw_extraction = False
        try:
            with self.client.post(
                self.analyze_path,
                data={
                    "text": final_transcript or SYNTHETIC_TRANSCRIPTS[language]["short"],
                    "language": language,
                    "mode": mode,
                    "department": self.department,
                    "use_history": "false",
                    "session_id": session_id,
                    "clinician_name": "locust-load-test",
                },
                name="POST /audio/analyze (ws-session)",
                stream=True,
                timeout=self.analyze_timeout,
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    failure_reason = failure_reason or f"HTTP {response.status_code}"
                else:
                    for raw_line in response.iter_lines(decode_unicode=False):
                        line = (raw_line or b"").decode("utf-8", errors="replace").strip()
                        if line.startswith("event:"):
                            name = line.removeprefix("event:").strip()
                            if not first_sse_event:
                                first_sse_event = name
                            if name == "extraction":
                                saw_extraction = True
                    if not saw_extraction:
                        failure_reason = failure_reason or "SSE stream ended without extraction"
                response.success() if not failure_reason else response.failure(failure_reason)
        except Exception as exc:
            failure_reason = failure_reason or _failure_category(exc)

        total_sec = round(time.perf_counter() - started, 3)
        _write_session_row(self.environment, {
            "session_id": session_id,
            "language": language,
            "mode": mode,
            "first_partial": first_partial,
            "final_transcript": final_transcript,
            "first_sse_event": first_sse_event,
            "total_sec": total_sec,
        })
        _fire_sse_metric(self.environment, "WS session total", total_sec * 1000)
