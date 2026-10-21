"""
tests/test_m2_adapters.py — Unit tests for M2.2 Phase 2 Concrete Provider Adapters
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from providers.base import LLMProvider, ProviderHealthResult, STTProvider
from providers.foundry_llm import FoundryLLMProvider
from providers.munsit_stt import MunsitSTTProvider
from providers.voxtral_stt import VoxtralSTTProvider


# ─────────────────────────────────────────────────────────────────────────────
# 1. Type hierarchy & ABC compliance
# ─────────────────────────────────────────────────────────────────────────────

def test_adapters_inherit_from_abcs():
    mock_runtime = MagicMock()
    foundry = FoundryLLMProvider(mock_runtime)
    munsit = MunsitSTTProvider(api_key="mock_key")
    voxtral = VoxtralSTTProvider(api_key="mock_key")

    assert isinstance(foundry, LLMProvider)
    assert isinstance(munsit, STTProvider)
    assert isinstance(voxtral, STTProvider)

    assert foundry.provider_name == "foundry"
    assert munsit.provider_name == "munsit"
    assert voxtral.provider_name == "voxtral"


# ─────────────────────────────────────────────────────────────────────────────
# 2. FoundryLLMProvider delegation & health checks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_foundry_complete_delegates_to_runtime():
    mock_runtime = MagicMock()
    mock_runtime.complete = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})
    mock_runtime.close = AsyncMock()

    provider = FoundryLLMProvider(mock_runtime)
    messages = [{"role": "user", "content": "Hello doctor"}]

    res = await provider.complete(
        model="gpt-4o",
        messages=messages,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    assert res == {"choices": [{"message": {"content": "ok"}}]}
    mock_runtime.complete.assert_awaited_once()
    call_kwargs = mock_runtime.complete.await_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert "create_kwargs" in call_kwargs
    create_kw = call_kwargs["create_kwargs"]
    assert create_kw["messages"] == messages

    await provider.close()
    mock_runtime.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_foundry_stream_delegates_to_runtime():
    mock_runtime = MagicMock()

    async def mock_stream_gen(model: str, create_kwargs: Dict[str, Any]) -> AsyncIterator[str]:
        yield "Chunk 1"
        yield "Chunk 2"

    mock_runtime.stream = mock_stream_gen

    provider = FoundryLLMProvider(mock_runtime)
    chunks = [
        c
        async for c in provider.stream(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Stream test"}],
            max_tokens=100,
        )
    ]

    assert chunks == ["Chunk 1", "Chunk 2"]


@pytest.mark.asyncio
async def test_foundry_health_check_success():
    # The provider adapter keeps its original lightweight models.list() probe.
    # The faster chat-completion probe used by GET /health lives in app.py
    # (_probe_foundry) so the adapter contract is unchanged.
    mock_runtime = MagicMock()
    mock_client = MagicMock()
    mock_client.models.list = AsyncMock(return_value=MagicMock())
    mock_runtime.client = mock_client

    provider = FoundryLLMProvider(mock_runtime)
    health = await provider.health_check()

    assert health.ok is True
    assert health.provider == "foundry"
    assert health.latency_ms is not None
    assert health.latency_ms >= 0


@pytest.mark.asyncio
async def test_foundry_health_check_failure():
    mock_runtime = MagicMock()
    mock_client = MagicMock()
    mock_client.models.list = AsyncMock(
        side_effect=RuntimeError("Connection refused")
    )
    mock_runtime.client = mock_client

    provider = FoundryLLMProvider(mock_runtime)
    health = await provider.health_check()

    assert health.ok is False
    assert health.provider == "foundry"
    assert "Connection refused" in (health.detail or "")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MunsitSTTProvider delegation & health checks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_munsit_transcribe_file_delegation():
    provider = MunsitSTTProvider(api_key="test_munsit_key")

    with patch("providers.munsit_stt.SpeechService.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
        mock_transcribe.return_value = "transcribed arabic audio text"

        result = await provider.transcribe_file("/tmp/fake_audio.wav", language="ar")
        assert result == "transcribed arabic audio text"
        mock_transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_munsit_transcribe_live_ar_delegation():
    provider = MunsitSTTProvider(api_key="test_munsit_key")

    async def fake_speech_live(*args, **kwargs):
        yield {"type": "interim", "text": "marhaban"}
        yield {"type": "final", "text": "marhaban bikum"}

    with patch("providers.munsit_stt.SpeechService.transcribe_live_stream_ar", side_effect=fake_speech_live):
        async def dummy_pcm():
            yield b"\x00\x00"

        deltas = [d async for d in provider.transcribe_live_ar(dummy_pcm())]
        assert deltas == ["marhaban bikum "]


@pytest.mark.asyncio
async def test_munsit_health_check():
    provider = MunsitSTTProvider(api_key="test_munsit_key")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        health = await provider.health_check()
        assert health.ok is True
        assert health.provider == "munsit"


# ─────────────────────────────────────────────────────────────────────────────
# 4. VoxtralSTTProvider delegation & health checks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voxtral_transcribe_file_delegation():
    provider = VoxtralSTTProvider(api_key="test_voxtral_key")

    with patch("providers.voxtral_stt.SpeechService.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
        mock_transcribe.return_value = "transcribed english audio text"

        result = await provider.transcribe_file("/tmp/fake_audio.wav", language="en")
        assert result == "transcribed english audio text"
        mock_transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_voxtral_transcribe_live_en_delegation():
    provider = VoxtralSTTProvider(api_key="test_voxtral_key")

    async def fake_speech_live(*args, **kwargs):
        yield "patient "
        yield "reports "
        yield "cough"

    with patch("providers.voxtral_stt.SpeechService.transcribe_live_stream", side_effect=fake_speech_live):
        async def dummy_pcm():
            yield b"\x00\x00"

        deltas = [d async for d in provider.transcribe_live_en(dummy_pcm())]
        assert deltas == ["patient ", "reports ", "cough"]


@pytest.mark.asyncio
async def test_voxtral_health_check():
    provider = VoxtralSTTProvider(api_key="test_voxtral_key")

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        health = await provider.health_check()
        assert health.ok is True
        assert health.provider == "voxtral"
