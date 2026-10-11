"""
Phase 2 — Application Insights Telemetry
=========================================
SDK: azure-monitor-opentelemetry  (NOT deprecated opencensus)
     This is the package Microsoft officially recommends as of 2024+.
     It ships OpenTelemetry + the Azure exporter in one install.

Three layers of instrumentation
---------------------------------
1. Auto-collection (configured once at startup via configure_azure_monitor):
     • Every HTTP request  → Request telemetry
     • Every exception     → Exception telemetry + stack trace
     • Python logging      → Trace telemetry (all logger.* calls forwarded)
     • asyncio tasks       → context propagation maintained automatically

2. Custom events (fire-and-forget, non-blocking):
     • AnalyzeSession  — one event per /audio/analyze pipeline run
     • DoctorFeedback  — one event per /feedback submission
     • VaultHealth     — emitted at startup to surface Key Vault state

3. Custom dependencies (fire-and-forget):
     • LLM calls       — model, latency, token counts, success/failure
     • STT calls       — provider, latency, success/failure
     • DB queries      — business unit, latency, success/failure

4. Custom metrics (gauges, non-blocking):
     • active_analyses      — concurrent /audio/analyze pipelines in flight
     • active_websockets    — concurrent /ws/transcribe sessions
     • llm_queue_depth      — semaphore backlog per model (from ModelLimiter)

Design decisions
-----------------
• ALL telemetry is fire-and-forget: every public function catches every
  exception and silently returns.  Telemetry must NEVER crash the app.
• Sensitive fields (transcript, refined_text, translated_text) are
  NEVER sent to App Insights — only metadata.
• JSONL files remain the primary audit log; App Insights is the analytics
  and alerting layer on top.
• When APPINSIGHTS_ENABLED=false (default), every function is a no-op
  — zero overhead and no SDK imports happen (deferred import pattern).

Environment variables
----------------------
  APPINSIGHTS_ENABLED                    = true | false (default: false)
  APPLICATIONINSIGHTS_CONNECTION_STRING  = InstrumentationKey=...;IngestionEndpoint=...
  APPINSIGHTS_CLOUD_ROLE                 = MedVoice-API (optional, labels the App Map node)
  APPINSIGHTS_SAMPLE_RATE               = 1.0 (optional, 0.0-1.0, default 1.0 = 100%)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime flags (read once at module import, after bootstrap())
# ---------------------------------------------------------------------------
_ENABLED: bool       = os.getenv("APPINSIGHTS_ENABLED", "false").lower() == "true"
_CONN_STRING: str    = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
_CLOUD_ROLE: str     = os.getenv("APPINSIGHTS_CLOUD_ROLE", "MedVoice-API")
_SAMPLE_RATE: float  = float(os.getenv("APPINSIGHTS_SAMPLE_RATE", "1.0"))

# Internal state — set by init_telemetry(), read-only afterwards
_initialized: bool = False
_tracer = None          # opentelemetry Tracer
_meter  = None          # opentelemetry Meter

# In-process metric gauges (thread-safe via asyncio event loop)
_active_analyses:   int = 0
_active_websockets: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_telemetry() -> None:
    """
    Initialize Application Insights. Called ONCE in app lifespan startup.

    Registers:
      • azure-monitor-opentelemetry auto-instrumentation (HTTP, exceptions, logs)
      • A named Tracer and Meter for manual spans / metrics

    No-op when APPINSIGHTS_ENABLED=false or connection string is missing.
    """
    global _initialized, _tracer, _meter

    if not _ENABLED:
        logger.info("[Telemetry] Application Insights disabled (APPINSIGHTS_ENABLED=false)")
        return

    if not _CONN_STRING:
        logger.warning(
            "[Telemetry] APPINSIGHTS_ENABLED=true but "
            "APPLICATIONINSIGHTS_CONNECTION_STRING is not set — disabled"
        )
        return

    if _initialized:
        logger.debug("[Telemetry] Already initialized — skipping")
        return

    try:
        # Azure Core logs full telemetry request/response details at INFO.
        # Keep warnings and errors visible without flooding container logs.
        logging.getLogger(
            "azure.core.pipeline.policies.http_logging_policy"
        ).setLevel(logging.WARNING)
        logging.getLogger(
            "azure.monitor.opentelemetry.exporter.export._base"
        ).setLevel(logging.WARNING)

        # ── 1. Auto-instrumentation: HTTP requests, exceptions, Python logs ─
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME

        configure_azure_monitor(
            connection_string=_CONN_STRING,
            resource=Resource.create({
                SERVICE_NAME:               _CLOUD_ROLE,
                "service.version":          "1.0.0",
                "cloud.role":               _CLOUD_ROLE,
            }),
            # Sampling: 1.0 = record every trace; lower for high-volume prod
            sampling_ratio=_SAMPLE_RATE,
            # Disable noisy SDK-level internal traces
            logger_name="azure",
        )

        # ── 2. Named Tracer + Meter for manual instrumentation ──────────────
        from opentelemetry import trace, metrics
        _tracer = trace.get_tracer("medvoice.api", "1.0.0")
        _meter  = metrics.get_meter("medvoice.api", "1.0.0")

        _initialized = True
        logger.info(
            "[Telemetry] Application Insights initialized — role=%s sample=%.0f%%",
            _CLOUD_ROLE, _SAMPLE_RATE * 100,
        )

    except ImportError:
        logger.warning(
            "[Telemetry] azure-monitor-opentelemetry not installed — "
            "run: pip install azure-monitor-opentelemetry>=1.6.0"
        )
    except Exception as exc:
        logger.error("[Telemetry] Init failed (app continues without telemetry): %s", exc)


# ---------------------------------------------------------------------------
# Custom Events
# ---------------------------------------------------------------------------

# Fields that must never be sent to App Insights (PHI / large blobs)
_REDACTED_FIELDS = frozenset({
    "transcript", "refined_text", "translated_text",
    "extracted_features", "questions", "diagnosis",
    "patient_story",
})


def track_event(
    name: str,
    properties: Optional[dict[str, Any]] = None,
    measurements: Optional[dict[str, float]] = None,
) -> None:
    """
    Send a custom event to Application Insights.

    Properties are string key-value pairs that appear in the App Insights
    "Events" blade and are queryable in KQL.  Measurements are numeric values
    that appear alongside the event for aggregation.

    Sensitive fields listed in _REDACTED_FIELDS are silently dropped so
    clinical text never leaves the VNet.

    Fire-and-forget: never raises, never blocks.
    """
    if not _ENABLED or not _initialized:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind

        # Sanitise properties: stringify values, drop sensitive fields
        safe_props: dict[str, str] = {}
        for k, v in (properties or {}).items():
            if k in _REDACTED_FIELDS:
                continue
            if v is None:
                continue
            safe_props[k] = str(v)

        # Emit as an OpenTelemetry span — Azure Monitor maps these to
        # custom events in the App Insights schema
        with _tracer.start_as_current_span(
            name,
            kind=SpanKind.INTERNAL,
            attributes={
                "event.name": name,
                **{f"event.{k}": v for k, v in safe_props.items()},
            },
        ) as span:
            # Attach numeric measurements as span attributes
            for k, v in (measurements or {}).items():
                if v is not None:
                    span.set_attribute(f"measurement.{k}", float(v))

        # Also log at DEBUG so it appears in Traces blade
        logger.debug(
            "[Telemetry] Event: %s props=%s measurements=%s",
            name, safe_props, measurements,
        )

    except Exception:
        pass  # Never fail on telemetry


# ---------------------------------------------------------------------------
# Custom Dependencies
# ---------------------------------------------------------------------------

def track_dependency(
    name: str,
    target: str,
    duration_ms: float,
    success: bool,
    data: str = "",
    dependency_type: str = "HTTP",
) -> None:
    """
    Record an outbound dependency call (LLM, STT, DB).

    Appears in App Insights → Application Map and Dependencies blade.
    Maps to OpenTelemetry CLIENT span with azure-monitor conventions.

    Args:
        name:            Human-readable call name (e.g. "LLM.complete")
        target:          Remote target (e.g. "openai/gpt-4o-mini")
        duration_ms:     Call duration in milliseconds
        success:         Whether the call succeeded
        data:            Optional detail (e.g. "tokens=512")
        dependency_type: "HTTP", "SQL", "gRPC", etc.
    """
    if not _ENABLED or not _initialized:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, StatusCode

        with _tracer.start_as_current_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                "peer.service":     target,
                "db.type":          dependency_type,
                "net.peer.name":    target,
                "dependency.data":  data,
            },
        ) as span:
            span.set_attribute("dependency.duration_ms", round(duration_ms, 1))
            if not success:
                span.set_status(StatusCode.ERROR, description=data)

    except Exception:
        pass  # Never fail on telemetry


# ---------------------------------------------------------------------------
# Custom Metrics
# ---------------------------------------------------------------------------

def track_metric(name: str, value: float, unit: str = "1") -> None:
    """
    Record a point-in-time metric value.

    Appears in App Insights → Metrics blade and can feed alerts.
    """
    if not _ENABLED or not _initialized:
        return

    try:
        # Use OTel meter if available
        if _meter is not None:
            gauge = _meter.create_gauge(name, unit=unit)
            gauge.set(value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Convenience: active-session counters (thread-safe via GIL on int ops)
# ---------------------------------------------------------------------------

def increment_active_analyses() -> None:
    """Call at the start of each /audio/analyze pipeline."""
    global _active_analyses
    _active_analyses += 1
    track_metric("medvoice.active_analyses", _active_analyses, unit="sessions")


def decrement_active_analyses() -> None:
    """Call at the end (success or failure) of each /audio/analyze pipeline."""
    global _active_analyses
    _active_analyses = max(0, _active_analyses - 1)
    track_metric("medvoice.active_analyses", _active_analyses, unit="sessions")


def increment_active_websockets() -> None:
    """Call at the start of each /ws/transcribe session."""
    global _active_websockets
    _active_websockets += 1
    track_metric("medvoice.active_websockets", _active_websockets, unit="sessions")


def decrement_active_websockets() -> None:
    """Call at the end of each /ws/transcribe session."""
    global _active_websockets
    _active_websockets = max(0, _active_websockets - 1)
    track_metric("medvoice.active_websockets", _active_websockets, unit="sessions")


def get_active_analyses() -> int:
    """Return the current number of in-flight /audio/analyze pipelines."""
    return _active_analyses


def get_active_websockets() -> int:
    """Return the current number of connected /ws/transcribe sessions."""
    return _active_websockets


# ---------------------------------------------------------------------------
# Context manager helpers for cleaner call-site code
# ---------------------------------------------------------------------------

class track_llm_call:
    """
    Context manager that measures LLM call duration and emits a dependency.

    Usage::

        async with track_llm_call(model="openai/gpt-4o-mini") as ctx:
            response = await runtime.complete(model=model, ...)
            ctx.tokens = getattr(response.usage, "total_tokens", None)
    """

    def __init__(self, model: str, operation: str = "complete") -> None:
        self.model     = model
        self.operation = operation
        self.tokens: Optional[int] = None
        self._t_start: float = 0.0
        self._success: bool  = True
        self._error: str     = ""

    def __enter__(self) -> "track_llm_call":
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self._t_start) * 1000
        if exc_type is not None:
            self._success = False
            self._error = str(exc_val)
        data = f"tokens={self.tokens}" if self.tokens is not None else self._error
        track_dependency(
            name=f"LLM.{self.operation}",
            target=self.model,
            duration_ms=duration_ms,
            success=self._success,
            data=data,
            dependency_type="HTTP",
        )

    # Support async with as well
    async def __aenter__(self) -> "track_llm_call":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class track_stt_call:
    """
    Context manager that measures STT call duration and emits a dependency.

    Usage::

        async with track_stt_call(provider="Munsit") as stt:
            result = await speech_service.transcribe(...)
    """

    def __init__(self, provider: str) -> None:
        self.provider  = provider
        self._t_start: float = 0.0
        self._success: bool  = True
        self._error: str     = ""

    def __enter__(self) -> "track_stt_call":
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self._t_start) * 1000
        if exc_type is not None:
            self._success = False
            self._error = str(exc_val)
        track_dependency(
            name="STT.transcribe",
            target=self.provider,
            duration_ms=duration_ms,
            success=self._success,
            data=self._error,
            dependency_type="HTTP",
        )

    async def __aenter__(self) -> "track_stt_call":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class track_db_call:
    """
    Context manager that measures DB query duration and emits a dependency.

    Usage::

        with track_db_call(business_unit="AHJ"):
            conn = _connect(business_unit)
            ...
    """

    def __init__(self, business_unit: str) -> None:
        self.business_unit = business_unit
        self._t_start: float = 0.0
        self._success: bool  = True
        self._error: str     = ""

    def __enter__(self) -> "track_db_call":
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self._t_start) * 1000
        if exc_type is not None:
            self._success = False
            self._error = str(exc_val)
        track_dependency(
            name="SQL.query",
            target=f"HCMS.{self.business_unit}",
            duration_ms=duration_ms,
            success=self._success,
            data=self._error,
            dependency_type="SQL",
        )
