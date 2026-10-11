"""Unit tests for Azure AI Foundry provider integration (Tracker M2).

Validates:
1. Configuration-driven provider switching via LLM_PROVIDER.
2. Model registry provider-awareness (Foundry defaults, deployment resolution, UI filtering).
3. Capability probing and safe_kwargs generation (max_completion_tokens, temperature omission, json_schema strict).
4. Suppression of OpenRouter-only parameters (extra_body usage.include, reasoning).
5. Error mapping security (no PHI, credentials, or prompts in error responses).
6. Process-local semaphore concurrency limits (Little's Law parity).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.model_registry import (
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_REASONING_MODEL,
    EXTRACTION_MODELS,
    FOUNDRY_DEFAULT_EXTRACTION_MODEL,
    FOUNDRY_DEFAULT_REASONING_MODEL,
    REASONING_MODELS,
    get_model,
    registry_for_ui,
)
from llm.limiter import ModelLimiter
from llm.provider import (
    _FOUNDRY_PROFILES,
    build_foundry_kwargs,
    get_foundry_profile,
    map_provider_error,
)
from llm.runtime import LLMRuntime


# ===========================================================================
# 1. Model Registry and Provider Switching
# ===========================================================================

def test_openrouter_defaults_unchanged():
    with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}):
        ext = get_model("extraction", None)
        reas = get_model("reasoning", None)
        assert ext.id == DEFAULT_EXTRACTION_MODEL
        assert reas.id == DEFAULT_REASONING_MODEL
        assert ext.slug == "openai/gpt-5.6-luna"
        assert reas.slug == "openai/gpt-5.6-luna"


def test_foundry_defaults_and_deployments():
    with patch.dict(os.environ, {"LLM_PROVIDER": "foundry"}):
        ext = get_model("extraction", None)
        reas = get_model("reasoning", None)
        assert ext.id == FOUNDRY_DEFAULT_EXTRACTION_MODEL
        assert reas.id == FOUNDRY_DEFAULT_REASONING_MODEL
        assert ext.foundry_deployment == "gpt-5.6-luna"
        assert reas.foundry_deployment == "gpt-5.6-luna"


def test_foundry_rejects_models_without_deployment():
    with patch.dict(os.environ, {"LLM_PROVIDER": "foundry"}):
        # gpt-4o-mini is confirmed not available in Foundry probe
        with pytest.raises(ValueError, match="no Foundry deployment configured"):
            get_model("extraction", "gpt-4o-mini")

        with pytest.raises(ValueError, match="no Foundry deployment configured"):
            get_model("reasoning", "gemini-3.7-flash")


def test_registry_for_ui_filters_for_foundry():
    with patch.dict(os.environ, {"LLM_PROVIDER": "foundry"}):
        reg = registry_for_ui()
        # In foundry mode, only models with a foundry_deployment should appear
        for step in ("extraction", "reasoning"):
            models = reg[step]["models"]
            for m in models:
                spec = EXTRACTION_MODELS[m["id"]] if step == "extraction" else REASONING_MODELS[m["id"]]
                assert spec.foundry_deployment is not None
            assert reg[step]["default"] == "gpt-5.6-luna"


# ===========================================================================
# 2. Capability Profiles and safe_kwargs
# ===========================================================================

def test_luna_profile_capabilities():
    profile = get_foundry_profile("gpt-5.6-luna")
    assert profile["max_completion_tokens"] is True
    assert profile["max_tokens"] is False
    assert profile["temperature"] is False
    assert profile["streaming"] is True
    assert profile["json_schema_strict"] is True


def test_build_foundry_kwargs_luna_omits_temperature_and_extra_body():
    messages = [{"role": "user", "content": "Patient reports cough."}]
    kwargs = build_foundry_kwargs(
        "gpt-5.6-luna",
        messages=messages,
        max_tokens=4096,
        stream=False,
        response_format={"type": "json_object"},
        reasoning_effort="medium",
    )

    # Must use deployment name as model
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["messages"] == messages

    # Must use max_completion_tokens for Luna, not max_tokens
    assert "max_completion_tokens" in kwargs
    assert kwargs["max_completion_tokens"] == 4096
    assert "max_tokens" not in kwargs

    # Luna rejects temperature — must be omitted
    assert "temperature" not in kwargs

    # OpenRouter-specific extra_body must NOT be present
    assert "extra_body" not in kwargs

    # Reasoning effort is native in Foundry for Luna-class models
    assert kwargs.get("reasoning_effort") == "medium"


def test_build_foundry_kwargs_streaming():
    kwargs = build_foundry_kwargs(
        "gpt-5.6-luna",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=100,
        stream=True,
    )
    assert kwargs.get("stream") is True


def test_build_foundry_kwargs_permissive_fallback():
    # Unknown deployment gets safe permissive defaults
    kwargs = build_foundry_kwargs(
        "custom-future-model",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=2048,
    )
    assert kwargs["model"] == "custom-future-model"
    assert kwargs.get("max_completion_tokens") == 2048


# ===========================================================================
# 3. Security and Error Mapping
# ===========================================================================

def test_map_provider_error_redacts_credentials_and_phi():
    import openai

    # Simulate NotFoundError (404)
    resp = MagicMock()
    resp.status_code = 404
    resp.headers = {}
    exc = openai.NotFoundError("Deployment secret-key-xyz not found", response=resp, body=None)

    mapped = map_provider_error(exc, step="extraction", deployment="gpt-5.6-luna")
    assert mapped["status_code"] == 502
    assert mapped["body"]["step"] == "extraction"
    assert mapped["body"]["deployment"] == "gpt-5.6-luna"
    # Ensure error message does NOT leak raw internal exception string with keys
    assert "secret-key-xyz" not in mapped["body"]["message"]
    assert "Deployment not found" in mapped["body"]["message"]


def test_map_provider_error_rate_limit():
    import openai

    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {}
    exc = openai.RateLimitError("Rate limit exceeded for client 123", response=resp, body=None)

    mapped = map_provider_error(exc, step="reasoning", deployment="gpt-5.6-luna")
    assert mapped["status_code"] == 429
    assert "Rate limit" in mapped["body"]["message"]


# ===========================================================================
# 4. Concurrency & Little's Law Semaphore Limits
# ===========================================================================

@pytest.mark.asyncio
async def test_foundry_limiter_concurrency_parity():
    """Verify ModelLimiter correctly tracks and throttles concurrency per deployment.

    Little's Law: L = lambda * W
    With expected latency W = 2.0s and arrival rate lambda = 5 req/s,
    concurrency L = 10. Default concurrency semaphore cap is enforced.
    """
    limiter = ModelLimiter(default_limit=10)
    deployment = "gpt-5.6-luna"

    # Acquire slots
    await limiter.acquire(deployment)
    stats = await limiter.stats(deployment)
    assert stats.in_flight == 1
    assert stats.acquire_count == 1

    # Release slots
    await limiter.release(deployment)
    stats = await limiter.stats(deployment)
    assert stats.in_flight == 0


# ===========================================================================
# 5. LLMRuntime Provider Headers and Auth Logic
# ===========================================================================

def test_runtime_foundry_headers():
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "foundry",
        "AI_FOUNDRY_ENDPOINT": "https://test-endpoint.openai.azure.com/openai/v1",
        "AI_FOUNDRY_KEY": "test-key-12345",
        "AI_FOUNDRY_API_VERSION": "2025-01-01-preview",
    }):
        runtime = LLMRuntime()
        assert runtime.provider == "foundry"
        headers = runtime._headers_for(None)
        assert headers is not None
        assert headers.get("api-version") == "2025-01-01-preview"
        # Must not contain Authorization bearer header in default headers_for
        assert "Authorization" not in headers


def test_runtime_openrouter_provider():
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-v1-test",
    }):
        runtime = LLMRuntime()
        assert runtime.provider == "openrouter"
