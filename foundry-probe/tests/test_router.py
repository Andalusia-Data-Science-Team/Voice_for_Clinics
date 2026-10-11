"""
tests/test_router.py – Unit tests for routing logic (no network, no cost).
Python 3.9 compatible.
"""
from __future__ import annotations

import os
import sys

# Make imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.router import evaluate_triggers, determine_initial_tier, determine_escalation

ROLES = {
    "TIER_SIMPLE":  "dep-luna",
    "TIER_DEFAULT": "dep-terra",
    "TIER_ESCALATE":"dep-sol",
    "STRUCTURING":  "dep-luna",
    "EXTRACTION":   "dep-mini",
}


def test_evaluate_triggers_problems():
    signals = {"problem_count": 4}
    triggers = evaluate_triggers(signals, "")
    assert "3+ active problems" in triggers


def test_evaluate_triggers_red_flags():
    signals = {"red_flags": ["chest pain"]}
    triggers = evaluate_triggers(signals, "")
    assert any("Red flags" in t for t in triggers)


def test_evaluate_triggers_history():
    signals = {}
    triggers = evaluate_triggers(signals, "patient has a history of diabetes")
    assert any("diabetes" in t for t in triggers)


def test_determine_initial_tier_simple():
    tier, _ = determine_initial_tier("difficulty", [], False, ROLES)
    assert tier == "dep-luna"


def test_determine_initial_tier_default():
    tier, _ = determine_initial_tier("difficulty", ["3+ active problems"], False, ROLES)
    assert tier == "dep-terra"


def test_determine_initial_tier_escalate():
    triggers = ["Red flags: chest pain", "3+ active problems"]
    tier, _ = determine_initial_tier("difficulty", triggers, False, ROLES)
    assert tier == "dep-sol"


def test_determine_initial_tier_v34_always_default():
    tier, _ = determine_initial_tier("v34", [], False, ROLES)
    assert tier == "dep-terra"


def test_determine_initial_tier_indepth_escalates():
    tier, _ = determine_initial_tier("difficulty", [], True, ROLES)
    assert tier == "dep-sol"


def test_escalation_v34_on_caution():
    esc, reason = determine_escalation("v34", "dep-terra", ["caution"], False, {}, ROLES)
    assert esc == "dep-sol"
    assert "Validation" in reason


def test_escalation_v34_no_caution():
    esc, _ = determine_escalation("v34", "dep-terra", [], False,
                                   {"final_diagnosis": "flu"}, ROLES)
    assert esc is None


def test_escalation_difficulty_from_simple():
    esc, _ = determine_escalation("difficulty", "dep-luna", ["caution"], False, {}, ROLES)
    assert esc == "dep-terra"


def test_escalation_difficulty_from_default():
    esc, _ = determine_escalation("difficulty", "dep-terra", ["caution"], False, {}, ROLES)
    assert esc == "dep-sol"


def test_no_escalation_no_cautions_difficulty():
    esc, _ = determine_escalation("difficulty", "dep-luna", [], False,
                                   {"final_diagnosis": "ok"}, ROLES)
    assert esc is None
