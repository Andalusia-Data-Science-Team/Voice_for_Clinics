"""
tests/test_providers_base.py — M2.1: Unit Tests for Provider ABC Contracts
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional
import pytest

from providers.base import LLMProvider, STTProvider, ProviderHealthResult


def test_provider_health_result_dataclass():
    res = ProviderHealthResult(ok=True, latency_ms=12.5, provider="test", detail="Healthy")
    assert res.ok is True
    assert res.latency_ms == 12.5
    assert res.provider == "test"
    assert res.detail == "Healthy"
    assert res.to_dict()["status"] == "ok"


def test_provider_health_result_convenience_constructors():
    h = ProviderHealthResult.healthy(provider="foundry", latency_ms=45.2)
    assert h.ok is True
    assert h.latency_ms == 45.2
    assert h.provider == "foundry"

    uh = ProviderHealthResult.unhealthy(provider="munsit", detail="timeout", latency_ms=5000.0)
    assert uh.ok is False
    assert uh.detail == "timeout"
    assert uh.to_dict()["status"] == "degraded"


def test_llm_provider_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore


def test_stt_provider_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        STTProvider()  # type: ignore


class ConcreteLLM(LLMProvider):
    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        return {"choices": [{"message": {"content": "completed"}}]}

    async def stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        yield "chunk 1"
        yield "chunk 2"

    async def close(self) -> None:
        pass

    async def health_check(self) -> ProviderHealthResult:
        return ProviderHealthResult.healthy(provider=self.provider_name, latency_ms=5.0)


class ConcreteSTT(STTProvider):
    async def transcribe_file(self, audio_path: str, language: str) -> str:
        return "transcribed file content"

    async def transcribe_live_ar(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        yield "arabic chunk"

    async def transcribe_live_en(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        yield "english chunk"

    async def health_check(self) -> ProviderHealthResult:
        return ProviderHealthResult.healthy(provider=self.provider_name, latency_ms=8.0)


@pytest.mark.asyncio
async def test_concrete_llm_implementation():
    llm = ConcreteLLM()
    assert llm.provider_name == "concrete"
    res = await llm.complete(model="test-model", messages=[{"role": "user", "content": "hi"}], max_tokens=100)
    assert res["choices"][0]["message"]["content"] == "completed"

    chunks = [c async for c in llm.stream(model="test-model", messages=[], max_tokens=10)]
    assert chunks == ["chunk 1", "chunk 2"]

    await llm.close()
    health = await llm.health_check()
    assert health.ok is True


@pytest.mark.asyncio
async def test_concrete_stt_implementation():
    stt = ConcreteSTT()
    assert stt.provider_name == "concrete"

    res = await stt.transcribe_file("/tmp/audio.wav", "ar")
    assert res == "transcribed file content"

    async def dummy_stream():
        yield b"chunk"

    chunks_ar = [c async for c in stt.transcribe_live_ar(dummy_stream())]
    assert chunks_ar == ["arabic chunk"]

    chunks_en = [c async for c in stt.transcribe_live_en(dummy_stream())]
    assert chunks_en == ["english chunk"]

    health = await stt.health_check()
    assert health.ok is True
