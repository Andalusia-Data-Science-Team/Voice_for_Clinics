"""
src/providers/base.py — M2.1: Provider Protocol Contracts

Defines the abstract base classes (ABCs) for the LLM and STT provider
abstraction layer. All concrete adapters (Phase 2) and all consumers
(llm_service.py, speech_service.py, app.py) MUST reference only these
interfaces — never concrete adapter classes.

Design notes:
  - Python 3.9-compatible: no X | Y unions, no match statements, no PEP 604.
  - Runtime enforcement via ABC: attempts to instantiate un-implemented
    subclasses raise TypeError, caught by tests/test_m2_contracts.py.
  - ProviderHealthResult is a frozen dataclass for immutability and
    equality semantics (useful in tests and health aggregation).
  - All async methods are declared with `async def` (not sync stubs)
    so type-checkers correctly infer coroutine / async-generator return types.
  - stream() and transcribe_live_* use AsyncIterator as the return type
    (not AsyncGenerator) to allow both `yield`-based and adapter-based
    implementations without coupling to a specific generator protocol.
  - PHI / security: no method signature accepts or returns patient text
    at this layer — that concern belongs in the concrete implementations.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Health result value object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderHealthResult:
    """Immutable result of a provider health / ping check.

    Attributes:
        ok:          True when the provider is reachable and responding.
        latency_ms:  Round-trip latency in milliseconds, or None if the probe
                     failed before a response was received.
        provider:    Human-readable provider name (e.g. "foundry", "munsit").
        detail:      Optional error message or extra context (never PHI).
        checked_at:  Monotonic timestamp (seconds) of when the check ran.
                     Populated automatically; pass 0.0 to override in tests.
    """

    ok: bool
    latency_ms: Optional[float] = None
    provider: str = "unknown"
    detail: Optional[str] = None
    checked_at: float = field(default_factory=time.monotonic)

    # ── Convenience constructors ──────────────────────────────────────────

    @classmethod
    def healthy(cls, provider: str, latency_ms: float) -> "ProviderHealthResult":
        """Return a passing health result."""
        return cls(ok=True, latency_ms=latency_ms, provider=provider)

    @classmethod
    def unhealthy(cls, provider: str, detail: str, latency_ms: Optional[float] = None) -> "ProviderHealthResult":
        """Return a failing health result with an error detail."""
        return cls(ok=False, latency_ms=latency_ms, provider=provider, detail=detail)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for use in /health and /admin/status JSON responses."""
        return {
            "ok": self.ok,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "status": "ok" if self.ok else "degraded",
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Provider ABC
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Uniform async interface for Large Language Model providers.

    CONCURRENCY CONTRACT: All implementations MUST be safe for concurrent use
    across many asyncio tasks in the same event loop.  Do NOT create a new
    HTTP client, SDK client, or connection pool per call — instantiate them
    once in __init__ or via a shared singleton.

    PROVIDER SWITCHING: Consumers must import and reference only this ABC, never
    a concrete implementation. The active implementation is chosen at startup by
    src/providers/factory.py based on the LLM_PROVIDER environment variable.

    SECURITY: No method in this interface should log, store, or expose secrets,
    API keys, PHI, or prompt content at the interface level. That concern belongs
    in concrete implementations.
    """

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        """Non-streaming chat completion.

        Args:
            model:            Provider deployment or model identifier.
            messages:         OpenAI-style list of {"role": ..., "content": ...} dicts.
            max_tokens:       Hard upper bound on output tokens.
            response_format:  Optional {"type": "json_object"} or {"type": "json_schema", ...}.
                              Implementations SHOULD downgrade gracefully if the model
                              does not support the requested format.
            reasoning_effort: Optional reasoning effort hint ("low", "medium", "high").
                              Ignored by models that do not support it.

        Returns:
            The raw SDK response object.  Callers extract content via
            ``response.choices[0].message.content``.

        Raises:
            Any provider-specific SDK exception — callers should use
            llm/provider.py::map_provider_error() to normalise errors.
        """

    @abstractmethod
    def stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Async generator yielding content chunk strings as they arrive.

        Args:
            model:           Provider deployment or model identifier.
            messages:        OpenAI-style message list.
            max_tokens:      Hard upper bound on output tokens.
            response_format: Optional response format hint (same as complete()).

        Yields:
            str: Individual content deltas (may be single characters or phrases).

        Raises:
            Any provider-specific SDK exception after yielding begins.

        Note:
            Implementations MUST NOT buffer all chunks before yielding — this
            method exists precisely to support progressive SSE streaming to the UI.
        """

    @abstractmethod
    async def close(self) -> None:
        """Gracefully release all connections and resources.

        Called by the FastAPI lifespan shutdown handler.  Implementations must
        be idempotent (safe to call multiple times) and must not raise.
        """

    @abstractmethod
    async def health_check(self) -> ProviderHealthResult:
        """Probe the provider endpoint with a cheap, non-PHI request.

        Used by GET /health and GET /admin/status.  Must complete quickly
        (implementations should apply an internal timeout of 3–5 s) and
        must never raise — failures should be returned as an unhealthy
        ProviderHealthResult, not as exceptions.

        Returns:
            ProviderHealthResult with ok=True if the provider is reachable.
        """

    # ── Non-abstract helpers ───────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier used in logs and health responses.

        Concrete implementations may override this; the default derives the
        name from the class name by stripping 'LLMProvider', 'LLM', or 'Provider'.
        """
        name = type(self).__name__
        for suffix in ("LLMProvider", "Provider", "LLM"):
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)]
                break
        return name.lower() or "llm"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} provider={self.provider_name!r}>"


# ─────────────────────────────────────────────────────────────────────────────
# STT Provider ABC
# ─────────────────────────────────────────────────────────────────────────────

class STTProvider(ABC):
    """Uniform async interface for Speech-to-Text providers.

    CONCURRENCY CONTRACT: Same as LLMProvider — implementations must be safe
    for concurrent use.  Do NOT open a new WebSocket connection per call in
    the non-streaming file transcription path; pool or reuse where possible.

    AUDIO FORMAT ASSUMPTION: All methods receive audio that has already been
    converted to pcm_s16le at 16 kHz mono (done by the audio pipeline in
    speech_service.py before reaching this interface).  Implementations must
    not do audio format conversion.

    PROVIDER SWITCHING: Chosen at startup via STT_PROVIDER env var.
    """

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    async def transcribe_file(
        self,
        audio_path: str,
        language: str,
    ) -> str:
        """Transcribe a complete, pre-recorded audio file.

        Args:
            audio_path: Absolute path to a local audio file.
            language:   BCP-47 language code, e.g. "ar", "en".

        Returns:
            Full transcript as a plain string.

        Raises:
            TranscriptionError (from speech_service) or provider-specific error.
        """

    @abstractmethod
    def transcribe_live_ar(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """Stream Arabic audio bytes → partial transcript chunks.

        Args:
            audio_stream: Async iterable of pcm_s16le bytes at 16 kHz.

        Yields:
            str: Partial or final transcript segments as they are recognised.

        Note:
            This method is the primary path for live clinical dictation in Arabic.
            Implementations must forward keepalive / silence packets as required
            by the underlying provider WebSocket protocol.
        """

    @abstractmethod
    def transcribe_live_en(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """Stream English audio bytes → partial transcript chunks.

        Args:
            audio_stream: Async iterable of pcm_s16le bytes at 16 kHz.

        Yields:
            str: Partial or final transcript segments as they are recognised.
        """

    @abstractmethod
    async def health_check(self) -> ProviderHealthResult:
        """Probe the STT endpoint with a lightweight, non-PHI request.

        Same contract as LLMProvider.health_check() — must not raise,
        must apply an internal timeout, must return ProviderHealthResult.
        """

    # ── Non-abstract helpers ───────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier."""
        name = type(self).__name__
        for suffix in ("STTProvider", "Provider", "STT"):
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)]
                break
        return name.lower() or "stt"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} provider={self.provider_name!r}>"


# ─────────────────────────────────────────────────────────────────────────────
# Public re-exports (consumers import from providers.base, not sub-modules)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ProviderHealthResult",
    "LLMProvider",
    "STTProvider",
]
