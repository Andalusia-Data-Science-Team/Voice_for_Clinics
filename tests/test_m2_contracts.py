"""
tests/test_m2_contracts.py — Unit tests for M2.1 ABC enforcement & contracts
"""
from __future__ import annotations

import inspect
import pytest

from providers.base import LLMProvider, ProviderHealthResult, STTProvider


def test_llm_provider_abstract():
    """Verify LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        LLMProvider()  # type: ignore
    assert "Can't instantiate abstract class" in str(exc_info.value)


def test_stt_provider_abstract():
    """Verify STTProvider cannot be instantiated directly."""
    with pytest.raises(TypeError) as exc_info:
        STTProvider()  # type: ignore
    assert "Can't instantiate abstract class" in str(exc_info.value)


def test_llm_provider_declared_abstract_methods():
    """Verify all expected abstract methods are enforced on LLMProvider."""
    abstracts = LLMProvider.__abstractmethods__
    assert "complete" in abstracts
    assert "stream" in abstracts
    assert "close" in abstracts
    assert "health_check" in abstracts


def test_stt_provider_declared_abstract_methods():
    """Verify all expected abstract methods are enforced on STTProvider."""
    abstracts = STTProvider.__abstractmethods__
    assert "transcribe_file" in abstracts
    assert "transcribe_live_ar" in abstracts
    assert "transcribe_live_en" in abstracts
    assert "health_check" in abstracts


def test_provider_health_result_immutability():
    """ProviderHealthResult must be immutable (frozen dataclass)."""
    res = ProviderHealthResult.healthy(provider="foundry", latency_ms=10.0)
    with pytest.raises(Exception):
        res.ok = False  # type: ignore
