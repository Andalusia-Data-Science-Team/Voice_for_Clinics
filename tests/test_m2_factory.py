"""
tests/test_m2_factory.py — Unit tests for M2.2 Phase 3 Provider Factory & Config Switch
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from providers.base import LLMProvider, STTProvider
from providers.factory import (
    build_llm_provider,
    build_stt_provider,
    get_llm,
    get_stt,
    reset_providers,
)
from providers.foundry_llm import FoundryLLMProvider
from providers.munsit_stt import MunsitSTTProvider
from providers.voxtral_stt import VoxtralSTTProvider


def setup_function():
    """Reset provider singletons before each test."""
    reset_providers()


def teardown_function():
    """Clean up provider singletons after each test."""
    reset_providers()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Zero-code environment variable switching
# ─────────────────────────────────────────────────────────────────────────────

def test_build_llm_provider_default_foundry():
    with patch.dict(os.environ, {"LLM_PROVIDER": "foundry"}):
        with patch("llm.runtime.LLMRuntime") as mock_runtime_cls:
            mock_runtime = MagicMock()
            mock_runtime_cls.return_value = mock_runtime

            provider = build_llm_provider()
            assert isinstance(provider, FoundryLLMProvider)
            assert isinstance(provider, LLMProvider)
            assert provider.provider_name == "foundry"


def test_build_llm_provider_unknown_raises_value_error():
    with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_vendor"}):
        with pytest.raises(ValueError) as exc_info:
            build_llm_provider()
        assert "Unknown LLM_PROVIDER" in str(exc_info.value)
        assert "unknown_vendor" in str(exc_info.value)


def test_build_stt_provider_munsit():
    with patch.dict(os.environ, {"STT_PROVIDER": "munsit", "MUNSIT_API_KEY": "test_munsit_key"}):
        provider = build_stt_provider()
        assert isinstance(provider, MunsitSTTProvider)
        assert isinstance(provider, STTProvider)
        assert provider.provider_name == "munsit"
        assert provider.api_key == "test_munsit_key"


def test_build_stt_provider_voxtral():
    with patch.dict(os.environ, {"STT_PROVIDER": "voxtral", "MISTRAL_API_KEY": "test_mistral_key"}):
        provider = build_stt_provider()
        assert isinstance(provider, VoxtralSTTProvider)
        assert isinstance(provider, STTProvider)
        assert provider.provider_name == "voxtral"
        assert provider.api_key == "test_mistral_key"


def test_build_stt_provider_unknown_raises_value_error():
    with patch.dict(os.environ, {"STT_PROVIDER": "invalid_engine"}):
        with pytest.raises(ValueError) as exc_info:
            build_stt_provider()
        assert "Unknown STT_PROVIDER" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Singleton identity (get_llm & get_stt)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_llm_singleton_identity():
    with patch.dict(os.environ, {"LLM_PROVIDER": "foundry"}):
        with patch("llm.runtime.LLMRuntime") as mock_runtime_cls:
            mock_runtime = MagicMock()
            mock_runtime_cls.return_value = mock_runtime

            inst1 = get_llm()
            inst2 = get_llm()
            assert inst1 is inst2
            assert mock_runtime_cls.call_count == 1


def test_get_stt_singleton_identity():
    with patch.dict(os.environ, {"STT_PROVIDER": "munsit", "MUNSIT_API_KEY": "dummy_key"}):
        inst1 = get_stt()
        inst2 = get_stt()
        assert inst1 is inst2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lifespan Integration Simulation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lifespan_closes_provider_cleanly():
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.close = AsyncMock()

    mock_stt = MagicMock(spec=STTProvider)

    with patch("providers.factory.build_llm_provider", return_value=mock_llm):
        with patch("providers.factory.build_stt_provider", return_value=mock_stt):
            from app import app, lifespan

            async with lifespan(app):
                assert app.state.llm_provider is mock_llm
                assert app.state.stt_provider is mock_stt
                assert app.state.llm is not None

            # Verify close called on shutdown
            mock_llm.close.assert_awaited_once()
