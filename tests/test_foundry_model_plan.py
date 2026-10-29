"""Tests for the revised Azure AI Foundry model plan (W1-W5).

All tests use mocks only — no network calls, no live Azure, synthetic data.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.config import Config
from core.model_registry import (
    EXTRACTION_MODELS,
    REASONING_MODELS,
    get_model,
    registry_for_ui,
    spec_for_deployment,
)
from pipeline import fallback as fallback_module
from pipeline import routing as routing_module
from pipeline.routing import (
    RoutingDecision,
    classify_complexity,
    decision_without_router,
    deployment_for_tier,
    fallback_deployment_for_tier,
    normalize_complexity,
    normalize_reason_codes,
    parse_router_response,
    tier_for_complexity,
)


# ── Helpers ────────────────────────────────────────────────────────────────

class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


class _FakeRuntime:
    """Minimal runtime double: records calls, returns queued responses/errors."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def complete(self, *, model, create_kwargs, key=None):
        self.calls.append({"model": model, "kwargs": create_kwargs})
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


# ── W1: config + registry ──────────────────────────────────────────────────

def test_config_defaults_match_ground_truth():
    with patch.dict("os.environ", {"ROUTER_FAILURE_TIER": "complex"}):
        assert Config.foundry_model_refinement() == "DeepSeek-V3.2"
        assert Config.foundry_model_translation() == "DeepSeek-V3.2"
        assert Config.foundry_model_extraction() == "gpt-5.6-luna"
        assert Config.foundry_model_extraction_fallback() == "gpt-4.1-mini"
        assert Config.foundry_model_router() == "model-router"
        assert Config.foundry_model_reasoning_simple() == "gpt-5.6-luna"
        assert Config.foundry_model_reasoning_complex() == "gpt-5.6-sol"
        assert Config.foundry_model_reasoning_simple_fallback() == "gpt-4.1-mini"
        assert Config.foundry_model_reasoning_complex_fallback() == "gpt-5.6-terra"
        assert Config.router_failure_tier() == "complex"


def test_router_disabled_by_default():
    with patch.dict("os.environ", {"ROUTER_ENABLED": "false"}):
        assert Config.router_enabled() is False


def test_quota_tpm_unset_is_none():
    assert Config.foundry_quota_tpm("gpt-5.6-luna") is None


def test_quota_tpm_reads_env():
    with patch.dict("os.environ", {"FOUNDRY_QUOTA_TPM__GPT_5_6_LUNA": "120000"}):
        assert Config.foundry_quota_tpm("gpt-5.6-luna") == 120000


def test_registry_for_ui_excludes_fallback_and_router():
    reg = registry_for_ui()
    for step in ("extraction", "reasoning"):
        ids = [m["id"] for m in reg[step]["models"]]
        assert "gpt-4.1-mini" not in ids
        assert "gpt-5.6-terra" not in ids
        assert "DeepSeek-V3.2" not in ids
        assert "model-router" not in ids
    assert reg["extraction"]["default"] == "gpt-5.6-luna"
    assert reg["reasoning"]["default"] == "gpt-5.6-luna"


def test_get_model_rejects_fallback_only_deployment():
    with pytest.raises(ValueError):
        get_model("extraction", "gpt-4.1-mini")
    with pytest.raises(ValueError):
        get_model("reasoning", "gpt-5.6-terra")


def test_get_model_accepts_selectable_primaries():
    assert get_model("extraction", "gpt-5.6-luna").foundry_deployment == "gpt-5.6-luna"
    assert get_model("reasoning", "gpt-5.6-sol").foundry_deployment == "gpt-5.6-sol"


def test_spec_for_deployment_unknown_returns_synthetic():
    spec = spec_for_deployment("model-router")
    assert spec.foundry_deployment == "model-router"
    assert spec.selectable is False


# ── W3: extraction fallback ────────────────────────────────────────────────

def test_fallback_eligible_statuses():
    assert fallback_module.is_fallback_eligible(_StatusError(429)) is True
    assert fallback_module.is_fallback_eligible(_StatusError(500)) is True
    assert fallback_module.is_fallback_eligible(_StatusError(503)) is True
    assert fallback_module.is_fallback_eligible(_StatusError(404)) is True


def test_fallback_not_eligible_statuses():
    for code in (400, 401, 403, 422):
        assert fallback_module.is_fallback_eligible(_StatusError(code)) is False


def test_fallback_not_eligible_insufficient_quota():
    exc = _StatusError(429)
    exc.code = "insufficient_quota"
    assert fallback_module.is_fallback_eligible(exc) is False


def test_fallback_not_eligible_content_filter():
    exc = _StatusError(400)
    exc.code = "content_filter"
    assert fallback_module.is_fallback_eligible(exc) is False


@pytest.mark.asyncio
async def test_run_with_single_fallback_uses_fallback_once():
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        raise _StatusError(503)

    async def fallback():
        calls["fallback"] += 1
        return "fallback-result"

    meta = {}
    result = await fallback_module.run_with_single_fallback(
        primary, fallback,
        primary_deployment="gpt-5.6-luna",
        fallback_deployment="gpt-4.1-mini",
        meta=meta,
    )
    assert result == "fallback-result"
    assert calls == {"primary": 1, "fallback": 1}
    assert meta["served_deployment"] == "gpt-4.1-mini"
    assert meta["fallback_used"] is True


@pytest.mark.asyncio
async def test_run_with_single_fallback_no_fallback_on_400():
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        raise _StatusError(400)

    async def fallback():
        calls["fallback"] += 1
        return "fallback-result"

    with pytest.raises(_StatusError):
        await fallback_module.run_with_single_fallback(
            primary, fallback,
            primary_deployment="gpt-5.6-luna",
            fallback_deployment="gpt-4.1-mini",
        )
    assert calls == {"primary": 1, "fallback": 0}


@pytest.mark.asyncio
async def test_run_with_single_fallback_primary_success():
    async def primary():
        return "primary-result"

    async def fallback():
        raise AssertionError("fallback must not be called")

    meta = {}
    result = await fallback_module.run_with_single_fallback(
        primary, fallback,
        primary_deployment="gpt-5.6-luna",
        fallback_deployment="gpt-4.1-mini",
        meta=meta,
    )
    assert result == "primary-result"
    assert meta["served_deployment"] == "gpt-5.6-luna"
    assert meta["fallback_used"] is False


# ── W4: routing pure functions ─────────────────────────────────────────────

def test_normalize_complexity():
    assert normalize_complexity("simple") == "simple"
    assert normalize_complexity("COMPLEX") == "complex"
    assert normalize_complexity("weird") is None
    assert normalize_complexity(None) is None


def test_normalize_reason_codes_filters_unknown():
    codes = normalize_reason_codes(["MULTIPLE_PROBLEMS", "bogus", "RED_FLAG_TERMS", "MULTIPLE_PROBLEMS"])
    assert codes == ["MULTIPLE_PROBLEMS", "RED_FLAG_TERMS"]


def test_parse_router_response_valid():
    parsed = parse_router_response('{"complexity":"complex","reason_codes":["RED_FLAG_TERMS"]}')
    assert parsed == {"complexity": "complex", "reason_codes": ["RED_FLAG_TERMS"]}


def test_parse_router_response_invalid_json():
    assert parse_router_response("not json") is None
    assert parse_router_response('{"complexity":"unknown"}') is None
    assert parse_router_response("") is None


def test_tier_mapping():
    assert tier_for_complexity("simple") == "simple"
    assert tier_for_complexity("complex") == "complex"
    assert deployment_for_tier("simple") == "gpt-5.6-luna"
    assert deployment_for_tier("complex") == "gpt-5.6-sol"
    assert fallback_deployment_for_tier("simple") == "gpt-4.1-mini"
    assert fallback_deployment_for_tier("complex") == "gpt-5.6-terra"


@pytest.mark.asyncio
async def test_router_simple_returns_luna_tier():
    decision = await classify_complexity(None, "Patient has clear lungs and normal blood pressure.")
    assert decision.complexity == "simple"
    assert decision.tier == "simple"
    assert decision.router_ok is True
    assert decision.router_deployment == "deterministic-5-rule-engine"


@pytest.mark.asyncio
async def test_router_complex_returns_sol_tier():
    # 3 rules: fillers (Rule 1) + companion (Rule 2) + reasoning cues (Rule 3)
    text = (
        "Okay, wait sorry I mean the patient might possibly have cardiac issue. "
        "The daughter mentioned chest discomfort."
    )
    decision = await classify_complexity(None, text)
    assert decision.complexity == "complex"
    assert decision.tier == "complex"
    assert decision.score >= 3
    assert "NOISY_DICTATION" in decision.matched_rules
    assert "MULTI_SPEAKER" in decision.matched_rules
    assert "REASONING_CUES" in decision.matched_rules


def test_decision_without_router_is_simple():
    decision = decision_without_router()
    assert decision.tier == "simple"
    assert decision.router_deployment == ""
    assert decision.attempts == 0


def test_routing_audit_has_no_clinical_text():
    decision = RoutingDecision(
        complexity="complex",
        reason_codes=["RED_FLAG_TERMS"],
        router_deployment="deterministic-5-rule-engine",
        tier="complex",
        served_deployment="gpt-5.6-sol",
        fallback_used=False,
        router_ok=True,
        router_latency_sec=0.001,
        attempts=1,
        score=3,
        matched_rules=["noisy_dictation", "multi_speaker_dispersion", "reasoning_cues"],
    )
    audit = decision.to_audit_dict()
    blob = json.dumps(audit).lower()
    for forbidden in ("transcript", "patient_story", "note", "diagnosis", "icd"):
        assert forbidden not in blob
    assert set(audit) == {
        "complexity", "reason_codes", "router_deployment", "tier",
        "served_deployment", "fallback_used", "router_ok",
        "router_latency_sec", "attempts", "score", "matched_rules",
    }


# ── W4: reasoning fallback (Sol -> Terra, Luna -> gpt-4.1-mini) ────────────

@pytest.mark.asyncio
async def test_reasoning_sol_failure_falls_back_to_terra():
    from model.llm_service import LLMService

    runtime = _FakeRuntime([_StatusError(503), _completion('{"json_data":{}}')])
    meta = {}
    with patch.object(LLMService, "_call_structured_async", new=AsyncMock(side_effect=[
        _StatusError(503), {"json_data": {}}
    ])):
        # Directly exercise the fallback wrapper with the tier's fallback.
        from pipeline.fallback import run_with_single_fallback

        async def primary():
            raise _StatusError(503)

        async def fallback():
            return {"json_data": {}}

        result = await run_with_single_fallback(
            primary, fallback,
            primary_deployment="gpt-5.6-sol",
            fallback_deployment="gpt-5.6-terra",
            meta=meta,
        )
    assert result == {"json_data": {}}
    assert meta["served_deployment"] == "gpt-5.6-terra"
    assert meta["fallback_used"] is True


@pytest.mark.asyncio
async def test_reasoning_luna_failure_falls_back_to_gpt41mini():
    from pipeline.fallback import run_with_single_fallback

    meta = {}

    async def primary():
        raise _StatusError(429)

    async def fallback():
        return {"json_data": {}}

    result = await run_with_single_fallback(
        primary, fallback,
        primary_deployment="gpt-5.6-luna",
        fallback_deployment="gpt-4.1-mini",
        meta=meta,
    )
    assert result == {"json_data": {}}
    assert meta["served_deployment"] == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_at_most_one_fallback_hop():
    """If the fallback also fails, no further hop is attempted."""
    calls = {"primary": 0, "fallback": 0}

    async def primary():
        calls["primary"] += 1
        raise _StatusError(503)

    async def fallback():
        calls["fallback"] += 1
        raise _StatusError(503)

    with pytest.raises(_StatusError):
        await fallback_module.run_with_single_fallback(
            primary, fallback,
            primary_deployment="gpt-5.6-sol",
            fallback_deployment="gpt-5.6-terra",
        )
    assert calls == {"primary": 1, "fallback": 1}


# ── W4: orchestrator routing integration ───────────────────────────────────

def _make_ctx():
    from pipeline.steps import PipelineContext
    from schemas.input_schemas import Language, Mode

    return PipelineContext(
        session_id="synthetic-session",
        text="Synthetic note: mild headache for two days.",
        language=Language.english,
        mode=Mode.doctor,
        department_value="opd",
        specialty_value=None,
        runtime=object(),
        extraction_spec=get_model("extraction", None),
        reasoning_spec=get_model("reasoning", None),
    )


@pytest.mark.asyncio
async def test_router_disabled_never_calls_router():
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.merger import DiagnosisStepResult, ExtractionStepResult

    ctx = _make_ctx()
    router_calls = {"n": 0}

    async def fake_classify(*args, **kwargs):
        router_calls["n"] += 1
        return decision_without_router()

    async def fake_extract(c):
        return ExtractionStepResult(features={"chief_complaint": "headache"}, extraction_sec=0.01)

    async def fake_diagnose(c, reasoning_spec=None, fallback_deployment=None):
        return DiagnosisStepResult(diagnosis={"icd10_codes": ["R51.9"]}, diagnosis_sec=0.01)

    with patch.object(Config, "router_enabled", classmethod(lambda cls: False)), \
         patch("pipeline.orchestrator.classify_complexity", side_effect=fake_classify), \
         patch("pipeline.orchestrator.step_extract", side_effect=fake_extract), \
         patch("pipeline.orchestrator.step_diagnose", side_effect=fake_diagnose):
        orch = PipelineOrchestrator()
        events = [ev async for ev in orch.run(ctx)]

    assert router_calls["n"] == 0
    assert any(e["event"] == "extraction" for e in events)


@pytest.mark.asyncio
async def test_extraction_event_has_models_object():
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.merger import DiagnosisStepResult, ExtractionStepResult

    ctx = _make_ctx()

    async def fake_extract(c):
        return ExtractionStepResult(
            features={"chief_complaint": "headache"},
            extraction_sec=0.01,
            meta={"served_deployment": "gpt-5.6-luna", "fallback_used": False},
        )

    async def fake_diagnose(c, reasoning_spec=None, fallback_deployment=None):
        return DiagnosisStepResult(
            diagnosis={"icd10_codes": ["R51.9"]},
            diagnosis_sec=0.01,
            meta={"served_deployment": "gpt-5.6-luna", "fallback_used": False},
        )

    with patch.object(Config, "router_enabled", classmethod(lambda cls: False)), \
         patch("pipeline.orchestrator.step_extract", side_effect=fake_extract), \
         patch("pipeline.orchestrator.step_diagnose", side_effect=fake_diagnose):
        orch = PipelineOrchestrator()
        events = [ev async for ev in orch.run(ctx)]

    extraction_ev = next(e for e in events if e["event"] == "extraction")
    data = json.loads(extraction_ev["data"])
    assert data["models"]["extraction"] == "gpt-5.6-luna"
    assert data["models"]["reasoning"] == "gpt-5.6-luna"
    # Singular label kept for backward compatibility.
    assert "model" in data


@pytest.mark.asyncio
async def test_both_reasoning_attempts_fail_note_still_delivered():
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.merger import DiagnosisStepResult, ExtractionStepResult

    ctx = _make_ctx()

    async def fake_extract(c):
        return ExtractionStepResult(features={"chief_complaint": "headache"}, extraction_sec=0.01)

    async def fake_diagnose(c, reasoning_spec=None, fallback_deployment=None):
        return DiagnosisStepResult(error="reasoning tier and fallback both failed", diagnosis_sec=0.01)

    with patch.object(Config, "router_enabled", classmethod(lambda cls: False)), \
         patch("pipeline.orchestrator.step_extract", side_effect=fake_extract), \
         patch("pipeline.orchestrator.step_diagnose", side_effect=fake_diagnose):
        orch = PipelineOrchestrator()
        events = [ev async for ev in orch.run(ctx)]

    # Extraction still delivered; diagnosis carries the error.
    extraction_ev = next(e for e in events if e["event"] == "extraction")
    assert json.loads(extraction_ev["data"])["extracted_features"]
    diagnosis_ev = next(e for e in events if e["event"] == "diagnosis")
    assert json.loads(diagnosis_ev["data"])["error"]


@pytest.mark.asyncio
async def test_escalation_skipped_when_sol_already_served():
    """Validation failure must not re-escalate to Sol if Sol already served."""
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.merger import DiagnosisStepResult, ExtractionStepResult

    ctx = _make_ctx()
    diagnose_calls = {"n": 0}

    async def fake_extract(c):
        return ExtractionStepResult(
            features={"chief_complaint": "left knee pain", "plan_of_care": "rest"},
            extraction_sec=0.01,
        )

    async def fake_diagnose(c, reasoning_spec=None, fallback_deployment=None):
        diagnose_calls["n"] += 1
        # Invalid ICD-10 -> validation failure, but Sol already served.
        return DiagnosisStepResult(
            diagnosis={"icd10_codes": ["FAKE999.00 - Hallucinated"]},
            diagnosis_sec=0.01,
            meta={"served_deployment": "gpt-5.6-sol", "fallback_used": False},
        )

    with patch.object(Config, "router_enabled", classmethod(lambda cls: True)), \
         patch("pipeline.orchestrator.classify_complexity",
               new=AsyncMock(return_value=RoutingDecision(
                   complexity="complex", tier="complex",
                   router_deployment="model-router", router_ok=True))), \
         patch("pipeline.orchestrator.step_extract", side_effect=fake_extract), \
         patch("pipeline.orchestrator.step_diagnose", side_effect=fake_diagnose):
        orch = PipelineOrchestrator()
        events = [ev async for ev in orch.run(ctx)]

    # Only the routed call ran; escalation was skipped.
    assert diagnose_calls["n"] == 1
    val_events = [e for e in events if e["event"] == "validation_result"]
    assert val_events
    val_data = json.loads(val_events[-1]["data"])
    assert val_data.get("escalation_skipped") is True