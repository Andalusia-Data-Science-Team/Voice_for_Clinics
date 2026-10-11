"""
Phase 4 — Admission Controller
===============================
Provides system-level overload protection and concurrency bounding for:
1. /audio/analyze pipelines (heavy CPU / LLM fanout)
2. /ws/transcribe live streaming sessions (memory / network sockets)

Integrates directly with Application Insights telemetry:
- Emits "AdmissionRejected" custom event when traffic exceeds capacity
- Records live concurrency metrics ("medvoice.active_analyses", "medvoice.active_websockets")
- Feeds the /health probe endpoint for Azure API Management and load balancers
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from core.telemetry import (
    decrement_active_analyses,
    decrement_active_websockets,
    get_active_analyses,
    get_active_websockets,
    increment_active_analyses,
    increment_active_websockets,
    track_event,
    track_metric,
)

logger = logging.getLogger(__name__)


class AdmissionRejectedError(Exception):
    """Raised when an incoming request is rejected due to capacity limits."""

    def __init__(self, path: str, reason: str, active: int, maximum: int) -> None:
        super().__init__(f"Capacity exceeded for {path}: {reason} (active={active}, max={maximum})")
        self.path = path
        self.reason = reason
        self.active = active
        self.maximum = maximum


class AdmissionController:
    """Manages system-level concurrency boundaries and health telemetry."""

    def __init__(self) -> None:
        self.max_analyses: int = int(os.getenv("MAX_CONCURRENT_ANALYSES", "20"))
        self.max_websockets: int = int(os.getenv("MAX_CONCURRENT_WEBSOCKETS", "50"))
        self._lock = asyncio.Lock()

    def reject(self, path: str, reason: str, active: int, maximum: int) -> None:
        """Record admission rejection in logs and Application Insights."""
        logger.warning(
            "[AdmissionControl] REJECTED path=%s reason=%s (active=%d, max=%d)",
            path, reason, active, maximum,
        )
        try:
            track_event(
                "AdmissionRejected",
                properties={
                    "path": path,
                    "reason": reason,
                    "active": str(active),
                    "max": str(maximum),
                },
                measurements={
                    "active": float(active),
                    "max": float(maximum),
                    "utilization_pct": float(active / maximum * 100.0) if maximum > 0 else 100.0,
                },
            )
            track_metric("medvoice.admission_rejections", 1.0, unit="events")
        except Exception:
            pass

    async def check_analysis(self) -> None:
        """Check if an /audio/analyze pipeline can be admitted."""
        active = get_active_analyses()
        if active >= self.max_analyses:
            self.reject("/audio/analyze", "max_analyses_reached", active, self.max_analyses)
            raise AdmissionRejectedError("/audio/analyze", "max_analyses_reached", active, self.max_analyses)

    async def check_websocket(self) -> None:
        """Check if a /ws/transcribe connection can be admitted."""
        active = get_active_websockets()
        if active >= self.max_websockets:
            self.reject("/ws/transcribe", "max_websockets_reached", active, self.max_websockets)
            raise AdmissionRejectedError("/ws/transcribe", "max_websockets_reached", active, self.max_websockets)

    @asynccontextmanager
    async def acquire_analysis_slot(self) -> AsyncIterator[None]:
        """Context manager to check admission and track active analyses lifetime."""
        await self.check_analysis()
        increment_active_analyses()
        try:
            yield
        finally:
            decrement_active_analyses()

    @asynccontextmanager
    async def acquire_websocket_slot(self) -> AsyncIterator[None]:
        """Context manager to check admission and track active websockets lifetime."""
        await self.check_websocket()
        increment_active_websockets()
        try:
            yield
        finally:
            decrement_active_websockets()

    @property
    def stats(self) -> Dict[str, Any]:
        """Return point-in-time concurrency and capacity metrics."""
        active_a = get_active_analyses()
        active_w = get_active_websockets()
        util_a = round(active_a / self.max_analyses, 3) if self.max_analyses > 0 else 1.0
        util_w = round(active_w / self.max_websockets, 3) if self.max_websockets > 0 else 1.0

        return {
            "active_analyses": active_a,
            "max_analyses": self.max_analyses,
            "analyses_utilization": util_a,
            "active_websockets": active_w,
            "max_websockets": self.max_websockets,
            "websockets_utilization": util_w,
            "is_saturated": (active_a >= self.max_analyses) or (active_w >= self.max_websockets),
        }


# Global singleton instance
admission_controller = AdmissionController()
