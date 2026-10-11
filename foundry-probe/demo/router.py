"""
demo/router.py – Routing logic.
Python 3.9 compatible. No hard-coded model names.
Model names come from the resolved _ROLES in app.py (passed in by caller).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "router_config.yaml")

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    CONFIG: Dict[str, Any] = yaml.safe_load(_f)


def evaluate_triggers(
    signals: Dict[str, Any],
    history_text: str,
) -> List[str]:
    triggers: List[str] = []
    thr = CONFIG.get("thresholds", {})

    problems = signals.get("problem_count", 0) or 0
    comorbs  = signals.get("comorbidity_count", 0) or 0
    meds     = signals.get("medication_count", 0) or 0
    spec_pop = (signals.get("special_population") or "none").lower()
    er_abn   = bool(signals.get("er_abnormal_primary_survey", False))
    red_flags: List[str] = signals.get("red_flags") or []

    if problems >= thr.get("active_problems", 3):
        triggers.append("3+ active problems")
    if comorbs >= thr.get("comorbidities", 2):
        triggers.append("2+ comorbidities")
    if meds >= thr.get("medications", 5):
        triggers.append("5+ medications")
    if spec_pop in ("pregnancy", "paediatric", "elderly"):
        triggers.append(f"Special population: {spec_pop}")
    if er_abn:
        triggers.append("ER abnormal primary survey")
    if red_flags:
        triggers.append(f"Red flags: {', '.join(red_flags)}")

    if history_text:
        hist_lower = history_text.lower()
        for hr in CONFIG.get("high_risk_diagnoses", []):
            if hr.lower() in hist_lower:
                triggers.append(f"History: {hr}")

    return triggers


def determine_initial_tier(
    mode: str,
    triggers: List[str],
    doctor_in_depth: bool,
    roles: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Optional[str], str]:
    """
    Returns (deployment_name, reason).
    roles: the resolved role map from app._ROLES.
    Falls back to None if a tier has no deployment.
    """
    if roles is None:
        roles = {}

    tier_simple   = roles.get("TIER_SIMPLE")
    tier_default  = roles.get("TIER_DEFAULT")
    tier_escalate = roles.get("TIER_ESCALATE")

    if mode == "v34":
        return tier_default, "v34 mode defaults to TIER_DEFAULT"

    # difficulty mode
    if doctor_in_depth:
        return tier_escalate, "Doctor requested in-depth review"

    has_red = any("Red flags" in t for t in triggers)
    has_3p  = any("3+ active problems" in t for t in triggers)
    has_spc = any("Special population" in t for t in triggers)
    has_er  = any("ER abnormal" in t for t in triggers)

    if has_red and (has_3p or has_spc or has_er):
        return tier_escalate, "Red flag AND complex condition → ESCALATE"
    if triggers:
        return tier_default, "Complexity trigger(s) → DEFAULT"
    return tier_simple, "No triggers → SIMPLE"


def determine_escalation(
    mode: str,
    initial_dep: Optional[str],
    cautions: List[str],
    doctor_in_depth: bool,
    reasoning_json: Dict[str, Any],
    roles: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Optional[str], str]:
    if roles is None:
        roles = {}

    tier_default  = roles.get("TIER_DEFAULT")
    tier_escalate = roles.get("TIER_ESCALATE")

    if mode == "v34":
        if doctor_in_depth:
            return tier_escalate, "Doctor in-depth → ESCALATE"
        if cautions:
            return tier_escalate, "Validation failed → ESCALATE"
        if not reasoning_json.get("final_diagnosis"):
            return tier_escalate, "No leading diagnosis → ESCALATE"
        return None, ""

    if mode == "difficulty":
        if cautions:
            if initial_dep == roles.get("TIER_SIMPLE"):
                return tier_default, "Validation failed → escalate to DEFAULT"
            if initial_dep == tier_default:
                return tier_escalate, "Validation failed → escalate to ESCALATE"
        return None, ""

    return None, ""


def log_audit(audit_data: Dict[str, Any]) -> None:
    os.makedirs("results", exist_ok=True)
    with open("results/audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_data, ensure_ascii=False) + "\n")
