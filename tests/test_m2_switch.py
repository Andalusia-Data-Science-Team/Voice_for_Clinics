"""
tests/test_m2_switch.py — Integration test for end-to-end provider switching
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from providers.factory import build_llm_provider, build_stt_provider, get_llm, reset_providers
from providers.foundry_llm import FoundryLLMProvider
from providers.munsit_stt import MunsitSTTProvider
from providers.voxtral_stt import VoxtralSTTProvider


def setup_function():
    reset_providers()


def teardown_function():
    reset_providers()


@pytest.mark.asyncio
async def test_llm_provider_switch_via_env():
    """Verify switching LLM_PROVIDER in env returns valid adapter without code changes."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "foundry"}):
        reset_providers()
        with patch("llm.runtime.LLMRuntime") as mock_runtime_cls:
            mock_runtime = MagicMock()
            mock_runtime.complete = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})
            mock_runtime_cls.return_value = mock_runtime

            provider = build_llm_provider()
            assert isinstance(provider, FoundryLLMProvider)
            result = await provider.complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=50,
            )
            assert result == {"choices": [{"message": {"content": "ok"}}]}


@pytest.mark.asyncio
async def test_stt_provider_switch_munsit_to_voxtral_via_env():
    """Verify switching STT_PROVIDER switches Arabic vs English engine seamlessly."""
    # 1. Switch to munsit
    with patch.dict(os.environ, {"STT_PROVIDER": "munsit", "MUNSIT_API_KEY": "munsit_123"}):
        reset_providers()
        provider_munsit = build_stt_provider()
        assert isinstance(provider_munsit, MunsitSTTProvider)
        assert provider_munsit.provider_name == "munsit"

    # 2. Switch to voxtral
    with patch.dict(os.environ, {"STT_PROVIDER": "voxtral", "MISTRAL_API_KEY": "mistral_123"}):
        reset_providers()
        provider_voxtral = build_stt_provider()
        assert isinstance(provider_voxtral, VoxtralSTTProvider)
        assert provider_voxtral.provider_name == "voxtral"


@pytest.mark.asyncio
async def test_app_health_endpoint_uses_provider_interface():
    """Verify app._check_llm executes through the provider interface."""
    from app import _check_llm
    from providers.base import ProviderHealthResult

    mock_provider = MagicMock()
    mock_provider.provider_name = "foundry"
    mock_provider.health_check = AsyncMock(
        return_value=ProviderHealthResult.healthy(provider="foundry", latency_ms=18.4)
    )

    mock_request = MagicMock()
    mock_request.app.state.llm_provider = mock_provider

    health_data = await _check_llm(mock_request)
    assert health_data["component"] == "llm"
    assert health_data["provider"] == "foundry"
    assert health_data["status"] == "ok"
    assert health_data["latency_ms"] == 18.4
    mock_provider.health_check.assert_awaited_once()
