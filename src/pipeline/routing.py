"""Reasoning routing step (W4).

The model-router deployment is used ONLY as a complexity classifier returning
strict JSON: {"complexity": "simple"|"complex", "reason_codes": [...]}.

Design:
  * Pure functions for the mapping/decision logic (easy to unit-test).
  * A thin async wrapper for the router call itself.
  * The router call uses the standard shared-client timeout (no separate
    router timeout) and never streams.

Flow (see pipeline.orchestrator):
  * ROUTER_ENABLED=false -> skip the router, reason on the simple tier at once.
  * ROUTER_ENABLED=true  -> run the router concurrently with extraction; when it
    returns, reason on the chosen tier.
  * Router failure / timeout / invalid JSON -> ROUTER_FAILURE_TIER (default complex).

The audit record attached to the usage log NEVER contains transcript, note or
any clinical text.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import Config
from core.model_registry import ModelSpec, spec_for_deployment
from llm.provider import build_foundry_kwargs
from llm.runtime import LLMRuntime
from utils.utils import safe_parse_json

logger = logging.getLogger(__name__)

ROUTING_CRITERIA_PATH = Path(__file__).resolve().parents[1] / "prompts" / "routing_criteria.txt"

VALID_COMPLEXITIES = ("simple", "complex")
VALID_REASON_CODES = (
    "MULTIPLE_PROBLEMS",
    "RED_FLAG_TERMS",
    "NOTE_HISTORY_CONFLICT",
    "MISSING_KEY_FIELDS",
)

ROUTER_MAX_TOKENS = 256


@dataclass
class RoutingDecision:
    """Outcome of the routing step. Contains NO clinical text."""

    complexity: str
    reason_codes: List[str] = field(default_factory=list)
    router_deployment: str = ""
    tier: str = "simple"
    served_deployment: str = ""
    fallback_used: bool = False
    router_ok: bool = False
    router_latency_sec: Optional[float] = None
    attempts: int = 0

    def to_audit_dict(self) -> Dict[str, Any]:
        """Return the audit record for the usage log (no clinical text)."""
        return {
            "complexity": self.complexity,
            "reason_codes": list(self.reason_codes),
            "router_deployment": self.router_deployment,
            "tier": self.tier,
            "served_deployment": self.served_deployment,
            "fallback_used": self.fallback_used,
            "router_ok": self.router_ok,
            "router_latency_sec": self.router_latency_sec,
            "attempts": self.attempts,
        }


# ── Pure decision functions ────────────────────────────────────────────────

def normalize_complexity(value: Any) -> Optional[str]:
    """Return 'simple'/'complex' for a valid value, else None."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in VALID_COMPLEXITIES:
            return v
    return None


def normalize_reason_codes(value: Any) -> List[str]:
    """Keep only known reason codes, preserving order and de-duplicating."""
    if not isinstance(value, list):
        return []
    seen: List[str] = []
    for item in value:
        if isinstance(item, str):
            code = item.strip().upper()
            if code in VALID_REASON_CODES and code not in seen:
                seen.append(code)
    return seen


def parse_router_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the router's strict-JSON response. Returns None when invalid."""
    if not raw or not raw.strip():
        return None
    try:
        parsed = safe_parse_json(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    complexity = normalize_complexity(parsed.get("complexity"))
    if complexity is None:
        return None
    return {
        "complexity": complexity,
        "reason_codes": normalize_reason_codes(parsed.get("reason_codes")),
    }


def tier_for_complexity(complexity: str) -> str:
    """Map a complexity label to a reasoning tier."""
    return "complex" if complexity == "complex" else "simple"


def deployment_for_tier(tier: str) -> str:
    """Return the primary deployment for a reasoning tier."""
    if tier == "complex":
        return Config.foundry_model_reasoning_complex()
    return Config.foundry_model_reasoning_simple()


def fallback_deployment_for_tier(tier: str) -> str:
    """Return the fallback deployment for a reasoning tier."""
    if tier == "complex":
        return Config.foundry_model_reasoning_complex_fallback()
    return Config.foundry_model_reasoning_simple_fallback()


def failure_tier() -> str:
    """Tier used when the router fails, times out, or returns invalid JSON."""
    return Config.router_failure_tier()


def load_routing_criteria() -> str:
    """Read the editable routing criteria file (DRAFT until signed off)."""
    try:
        return ROUTING_CRITERIA_PATH.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - file is shipped with the repo
        logger.warning("[Routing] Could not read routing criteria: %s", exc)
        return ""


def build_router_prompt(english_text: str, patient_history_text: str = "") -> str:
    """Build the router prompt. The text is used only for classification."""
    criteria = load_routing_criteria()
    history_block = patient_history_text.strip() or "(none provided)"
    return (
        "You are a complexity classifier for a clinical reasoning step.\n"
        "Classify the following clinical note as 'simple' or 'complex' using the "
        "criteria below. Return ONLY strict JSON of the form "
        '{"complexity":"simple"|"complex","reason_codes":[...]}.\n\n'
        "=== CRITERIA ===\n"
        f"{criteria}\n"
        "=== PATIENT HISTORY ===\n"
        f"{history_block}\n"
        "=== CLINICAL NOTE ===\n"
        f"{english_text}\n"
    )


# ── Thin async wrapper ─────────────────────────────────────────────────────

async def classify_complexity(
    runtime: LLMRuntime,
    english_text: str,
    patient_history_text: str = "",
) -> RoutingDecision:
    """Call the router deployment and return a RoutingDecision.

    Never raises: on any failure the decision falls back to ROUTER_FAILURE_TIER.
    Uses the standard shared-client timeout (no separate router timeout).
    """
    router_deployment = Config.foundry_model_router()
    decision = RoutingDecision(
        complexity=failure_tier(),
        router_deployment=router_deployment,
        tier=tier_for_complexity(failure_tier()),
        attempts=1,
    )
    prompt = build_router_prompt(english_text, patient_history_text)
    kwargs = build_foundry_kwargs(
        router_deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=ROUTER_MAX_TOKENS,
        stream=False,
        response_format={"type": "json_object"},
    )
    t0 = time.perf_counter()
    try:
        response = await runtime.complete(model=router_deployment, create_kwargs=kwargs)
        raw = ""
        if response.choices and response.choices[0].message.content:
            raw = response.choices[0].message.content
        parsed = parse_router_response(raw)
        if parsed is None:
            logger.warning("[Routing] Router returned invalid JSON; using failure tier=%s", failure_tier())
            decision.router_ok = False
        else:
            decision.complexity = parsed["complexity"]
            decision.reason_codes = parsed["reason_codes"]
            decision.tier = tier_for_complexity(parsed["complexity"])
            decision.router_ok = True
    except Exception as exc:
        logger.warning("[Routing] Router call failed (%s); using failure tier=%s", type(exc).__name__, failure_tier())
        decision.router_ok = False
    finally:
        decision.router_latency_sec = round(time.perf_counter() - t0, 3)
    return decision


def decision_without_router() -> RoutingDecision:
    """Return the decision used when ROUTER_ENABLED=false (simple tier, no call)."""
    tier = "simple"
    return RoutingDecision(
        complexity="simple",
        reason_codes=[],
        router_deployment="",
        tier=tier,
        served_deployment=deployment_for_tier(tier),
        fallback_used=False,
        router_ok=False,
        router_latency_sec=None,
        attempts=0,
    )


def spec_for_tier(tier: str) -> ModelSpec:
    """Return the ModelSpec for a reasoning tier's primary deployment."""
    return spec_for_deployment(deployment_for_tier(tier), supports_reasoning=True)


__all__ = [
    "RoutingDecision",
    "classify_complexity",
    "decision_without_router",
    "deployment_for_tier",
    "fallback_deployment_for_tier",
    "failure_tier",
    "normalize_complexity",
    "normalize_reason_codes",
    "parse_router_response",
    "spec_for_tier",
    "tier_for_complexity",
]