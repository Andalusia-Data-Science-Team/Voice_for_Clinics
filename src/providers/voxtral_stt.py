"""
src/providers/voxtral_stt.py — M2.2: Concrete Voxtral STT Provider Adapter

Wraps Voxtral/Mistral Realtime STT for English transcription and file transcription.
Delegates to SpeechService under the hood to preserve existing pipeline logic.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Optional

import httpx

from core.config import Config
from model.speech_service import SpeechService
from providers.base import ProviderHealthResult, STTProvider


class VoxtralSTTProvider(STTProvider):
    """Voxtral (Mistral) implementation of STTProvider."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or Config.MISTRAL_API_KEY

    @property
    def api_key(self) -> str:
        return self._api_key or ""

    async def transcribe_file(
        self,
        audio_path: str,
        language: str = "en",
    ) -> str:
        """Transcribe an audio file using Mistral/Voxtral batch transcription."""
        res = await SpeechService.transcribe_audio(
            audio_file_path=audio_path,
            api_key=self.api_key,
            language=language,
        )
        if isinstance(res, tuple):
            return res[0]
        return res

    async def transcribe_live_ar(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """Delegate Arabic stream to Munsit if needed or through speech service."""
        munsit_key = Config.MUNSIT_API_KEY or ""
        async for event in SpeechService.transcribe_live_stream_ar(
            raw_pcm_iter=audio_stream,
            api_key=munsit_key,
            code_switching=True,
        ):
            if event.get("type") == "final":
                text = event.get("text", "")
                if text:
                    yield text + " "

    async def transcribe_live_en(
        self,
        audio_stream: AsyncIterator[bytes],
        input_sample_rate: int = 16000,
    ) -> AsyncIterator[str]:
        """Stream English audio bytes to Voxtral and yield text deltas in real time."""
        async for delta in SpeechService.transcribe_live_stream(
            raw_pcm_iter=audio_stream,
            api_key=self.api_key,
            input_sample_rate=input_sample_rate,
        ):
            yield delta

    async def health_check(self) -> ProviderHealthResult:
        """Probe Mistral/Voxtral API endpoint reachable via HTTPS."""
        start = time.perf_counter()
        try:
            url = "https://api.mistral.ai/v1/models"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                latency = round((time.perf_counter() - start) * 1000.0, 2)
                if resp.status_code in (200, 401, 403):
                    return ProviderHealthResult.healthy(
                        provider=self.provider_name,
                        latency_ms=latency,
                    )
                return ProviderHealthResult.unhealthy(
                    provider=self.provider_name,
                    detail=f"Mistral HTTP status {resp.status_code}",
                    latency_ms=latency,
                )
        except Exception as exc:
            latency = round((time.perf_counter() - start) * 1000.0, 2)
            return ProviderHealthResult.unhealthy(
                provider=self.provider_name,
                detail=str(exc),
                latency_ms=latency,
            )


__all__ = ["VoxtralSTTProvider"]
