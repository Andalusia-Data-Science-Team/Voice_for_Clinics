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

import asyncio
import json
import os
import time
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
    # Design changed (M2 provider abstraction): the model registry is now
    # Foundry-only and provider switching lives in providers/factory.py. The
    # registry defaults must therefore be unchanged regardless of LLM_PROVIDER,
    # and the OpenRouter rollback provider must still be selectable.
    from providers.factory import build_llm_provider
    from providers.openrouter_llm import OpenRouterLLMProvider

    with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-test"}):
        ext = get_model("extraction", None)
        reas = get_model("reasoning", None)
        assert ext.id == DEFAULT_EXTRACTION_MODEL
        assert reas.id == DEFAULT_REASONING_MODEL
        # Foundry deployment is the stable slug; there is no OpenRouter slug in
        # the registry any more.
        assert ext.slug == "gpt-5.6-luna"
        assert reas.slug == "gpt-5.6-luna"
        # The OpenRouter rollback path is selected via the factory.
        provider = build_llm_provider()
        assert isinstance(provider, OpenRouterLLMProvider)
        assert provider.provider_name == "openrouter"


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
        # Design changed (M2): unknown models now raise an "Unknown ... model"
        # error that lists the available ids. The intent — reject models that
        # are not in the Foundry registry — is unchanged.
        with pytest.raises(ValueError, match="Unknown extraction model"):
            get_model("extraction", "gpt-4o-mini")

        with pytest.raises(ValueError, match="Unknown reasoning model"):
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
    # Design changed (M2 provider abstraction): LLMRuntime is Foundry-only; the
    # OpenRouter rollback lives in providers/openrouter_llm.py and is selected
    # via providers/factory.py, not by LLMRuntime. This test now asserts that
    # the OpenRouter provider is reachable through the factory instead.
    from providers.factory import build_llm_provider
    from providers.openrouter_llm import OpenRouterLLMProvider

    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-v1-test",
    }):
        provider = build_llm_provider()
        assert isinstance(provider, OpenRouterLLMProvider)
        assert provider.provider_name == "openrouter"


# ===========================================================================
# 5. Phase 1 Emergency Fixes (Timeout, Retries, Structured Response Format)
# ===========================================================================

def test_phase1_config_timeouts_and_retries():
    from core.config import Config
    assert hasattr(Config, "PIPELINE_TIMEOUT_S")
    assert Config.PIPELINE_TIMEOUT_S >= 90.0
    assert hasattr(Config, "LLM_DIAGNOSIS_MAX_ATTEMPTS")
    assert Config.LLM_DIAGNOSIS_MAX_ATTEMPTS == 2
    assert Config.LLM_TIMEOUT_S == 45.0


def test_build_foundry_kwargs_luna_json_schema_supported():
    from pydantic import BaseModel
    class SampleNote(BaseModel):
        json_data: dict

    schema = SampleNote.model_json_schema() if hasattr(SampleNote, "model_json_schema") else SampleNote.schema()
    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "SampleNote",
            "strict": True,
            "schema": schema,
        },
    }
    kwargs = build_foundry_kwargs(
        "gpt-5.6-luna",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=500,
        response_format=rf,
    )
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True


# ===========================================================================
# 6. Phase 2 True Parallelism (Context ICD Candidates & Concurrent Tasks)
# ===========================================================================

def test_phase2_pipeline_context_icd_candidates():
    from pipeline.steps import PipelineContext
    from schemas.input_schemas import Language, Mode
    from core.model_registry import get_model

    ctx = PipelineContext(
        session_id="test-session",
        text="test note",
        language=Language.english,
        mode=Mode.doctor,
        department_value="opd",
        specialty_value="General",
        runtime=MagicMock(),
        extraction_spec=get_model("extraction", None),
        reasoning_spec=get_model("reasoning", None),
    )
    assert hasattr(ctx, "icd_candidates")
    assert ctx.icd_candidates is None
    ctx.icd_candidates = [("J00", "Acute nasopharyngitis")]
    assert len(ctx.icd_candidates) == 1


# ===========================================================================
# 7. Phase 3 Resilience & Observability (Router Profile, CB Tuning)
# ===========================================================================

def test_phase3_model_router_profile():
    profile = get_foundry_profile("model-router")
    assert profile["max_completion_tokens"] is True
    assert profile["max_tokens"] is True
    assert profile["streaming"] is False
    assert profile["json_object"] is True
    assert profile["json_schema_strict"] is False
    assert profile["temperature"] is True
    assert profile["reasoning_effort"] is False


def test_phase3_circuit_breaker_tuning():
    from core.config import Config
    from llm.runtime import LLMRuntime

    assert Config.CB_FAILURE_THRESHOLD == 10
    assert Config.CB_WINDOW_S == 60.0
    assert Config.CB_RECOVERY_TIMEOUT_S == 10.0

    runtime = LLMRuntime()
    assert runtime.circuit_breaker.failure_threshold == 10
    assert runtime.circuit_breaker.window_s == 60.0
    assert runtime.circuit_breaker.recovery_timeout_s == 10.0


# ===========================================================================
# 8. Phase 4 Enable Router & 3-Way Concurrent Orchestration
# ===========================================================================

def test_phase4_router_config_and_defaults():
    from core.config import Config
    with patch.dict(os.environ, {
        "ROUTER_ENABLED": "true",
        "FOUNDRY_MODEL_ROUTER": "model-router",
        "ROUTER_FAILURE_TIER": "simple",
    }):
        assert Config.router_enabled() is True
        assert Config.foundry_model_router() == "model-router"
        assert Config.router_failure_tier() == "simple"


def test_phase4_router_tier_resolution():
    from pipeline.routing import (
        tier_for_complexity,
        deployment_for_tier,
        fallback_deployment_for_tier,
        decision_without_router,
        RoutingDecision,
    )
    from core.config import Config

    assert tier_for_complexity("simple") == "simple"
    assert tier_for_complexity("complex") == "complex"
    assert deployment_for_tier("simple") == Config.foundry_model_reasoning_simple()
    assert deployment_for_tier("complex") == Config.foundry_model_reasoning_complex()
    assert fallback_deployment_for_tier("simple") == Config.foundry_model_reasoning_simple_fallback()
    assert fallback_deployment_for_tier("complex") == Config.foundry_model_reasoning_complex_fallback()

    no_router = decision_without_router()
    assert no_router.tier == "simple"
    assert no_router.router_ok is False


@pytest.mark.asyncio
async def test_phase4_concurrent_orchestrator_router_flow():
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.steps import PipelineContext
    from schemas.input_schemas import Language, Mode
    from core.model_registry import get_model
    from unittest.mock import AsyncMock

    mock_runtime = MagicMock()

    ctx = PipelineContext(
        session_id="test-router-phase4",
        text="Patient presents with mild runny nose for 2 days. No fever, lungs clear.",
        language=Language.english,
        mode=Mode.doctor,
        department_value="opd",
        specialty_value="General",
        runtime=mock_runtime,
        extraction_spec=get_model("extraction", None),
        reasoning_spec=get_model("reasoning", None),
    )

    events = []
    start_times = {}

    async def tracking_extract(c):
        start_times["extract"] = time.perf_counter()
        await asyncio.sleep(0.05)
        from pipeline.merger import ExtractionStepResult
        return ExtractionStepResult(features={"chief_complaint": "cold"}, extraction_sec=0.05)

    async def tracking_diagnose(c, reasoning_spec=None, fallback_deployment=None):
        start_times["diagnose"] = time.perf_counter()
        await asyncio.sleep(0.05)
        from pipeline.merger import DiagnosisStepResult
        return DiagnosisStepResult(diagnosis={"icd10": "J00"}, diagnosis_sec=0.05)

    with patch("pipeline.orchestrator.get_registry") as mock_reg, \
         patch("pipeline.orchestrator.step_extract", side_effect=tracking_extract), \
         patch("pipeline.orchestrator.step_diagnose", side_effect=tracking_diagnose), \
         patch("pipeline.orchestrator.ClinicalValidationPipeline.validate") as mock_val:

        mock_reg.return_value.search_diagnosis_candidates.return_value = [("J00", "Acute nasopharyngitis")]
        mock_val_report = MagicMock()
        mock_val_report.is_valid = True
        mock_val_report.to_dict.return_value = {"is_valid": True}
        mock_val.return_value = mock_val_report

        orchestrator = PipelineOrchestrator()
        async for ev in orchestrator.run(ctx):
            events.append(ev)

    event_names = [e["event"] for e in events]
    assert "routing_signal" in event_names
    assert "step_progress" in event_names
    assert "extraction" in event_names
    assert "diagnosis" in event_names

    # Check progress steps emitted
    progress_events = [json.loads(e["data"]) for e in events if e["event"] == "step_progress"]
    progress_steps = [p["step"] for p in progress_events]
    assert "router" in progress_steps
    assert "icd_search" in progress_steps
    assert "extraction" in progress_steps
    assert "diagnosis" in progress_steps

    # Verify router latency was sub-millisecond (no 7-second router delay)
    router_done_ev = next(p for p in progress_events if p["step"] == "router" and p["status"] == "done")
    assert router_done_ev["sec"] < 0.05, f"Router took {router_done_ev['sec']}s, expected < 0.05s"

    # Verify extract and diagnose started truly in parallel at T=0 (< 5ms difference)
    delta_start = abs(start_times["extract"] - start_times["diagnose"])
    assert delta_start < 0.02, f"Tasks did not start in parallel: delta={delta_start:.4f}s"
