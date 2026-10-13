"""
src/providers/munsit_stt.py — M2.2: Concrete Munsit STT Provider Adapter

Wraps Munsit STT live-streaming and batch file transcription.
Delegates to SpeechService under the hood to preserve existing pipeline logic.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Dict, Optional

import httpx

from core.config import Config
from model.speech_service import SpeechService
from providers.base import ProviderHealthResult, STTProvider


class MunsitSTTProvider(STTProvider):
    """Munsit implementation of STTProvider for Arabic and code-switching clinical dictation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or Config.MUNSIT_API_KEY

    @property
    def api_key(self) -> str:
        return self._api_key or ""

    async def transcribe_file(
        self,
        audio_path: str,
        language: str = "ar",
    ) -> str:
        """Transcribe an audio file using speech service.

        Falls back to SpeechService.transcribe_audio if language is handled by it.
        """
        mistral_key = Config.MISTRAL_API_KEY or ""
        res = await SpeechService.transcribe_audio(
            audio_file_path=audio_path,
            api_key=mistral_key,
            language=language,
        )
        if isinstance(res, tuple):
            return res[0]
        return res

    async def transcribe_live_ar(
        self,
        audio_stream: AsyncIterator[bytes],
        input_sample_rate: int = 16000,
        hotwords: Optional[str] = None,
        code_switching: bool = True,
    ) -> AsyncIterator[str]:
        """Stream Arabic audio bytes to Munsit and yield normalized text deltas."""
        async for event in SpeechService.transcribe_live_stream_ar(
            raw_pcm_iter=audio_stream,
            api_key=self.api_key,
            input_sample_rate=input_sample_rate,
            hotwords=hotwords,
            code_switching=code_switching,
        ):
            etype = event.get("type")
            if etype == "final":
                text = event.get("text", "")
                if text:
                    yield text + " "
            elif etype == "interim":
                # For basic string stream, optionally yield interim or skip
                continue

    async def transcribe_live_en(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """Stream English audio bytes — delegate to live stream if needed."""
        mistral_key = Config.MISTRAL_API_KEY or ""
        async for delta in SpeechService.transcribe_live_stream(
            raw_pcm_iter=audio_stream,
            api_key=mistral_key,
        ):
            yield delta

    async def health_check(self) -> ProviderHealthResult:
        """Probe Munsit service reachable via HTTPS check.

        Uses the voices endpoint or base API with api_key.
        """
        start = time.perf_counter()
        try:
            url = "https://api.munsit.com/api/v1/voices"
            headers = {"x-api-key": self.api_key} if self.api_key else {}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                latency = round((time.perf_counter() - start) * 1000.0, 2)
                # 200 or 401/403 means network reachable
                if resp.status_code in (200, 401, 403):
                    return ProviderHealthResult.healthy(
                        provider=self.provider_name,
                        latency_ms=latency,
                    )
                return ProviderHealthResult.unhealthy(
                    provider=self.provider_name,
                    detail=f"Munsit HTTP status {resp.status_code}",
                    latency_ms=latency,
                )
        except Exception as exc:
            latency = round((time.perf_counter() - start) * 1000.0, 2)
            return ProviderHealthResult.unhealthy(
                provider=self.provider_name,
                detail=str(exc),
                latency_ms=latency,
            )


__all__ = ["MunsitSTTProvider"]
