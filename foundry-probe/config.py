"""
config.py – Central configuration and shared AsyncOpenAI client.
Python 3.9 compatible.
SECURITY: keys only from env, never printed or logged.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── Required ──────────────────────────────────────────────────────────────────
FOUNDRY_API_KEY: str = os.getenv("FOUNDRY_API_KEY", "")
FOUNDRY_BASE_URL: str = os.getenv("FOUNDRY_BASE_URL", "")
FOUNDRY_PROJECT_ENDPOINT: str = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")

if not FOUNDRY_API_KEY or not FOUNDRY_BASE_URL:
    raise ValueError(
        "Missing FOUNDRY_API_KEY or FOUNDRY_BASE_URL. "
        "Set them in .env (never hard-code)."
    )

# ── Optional ──────────────────────────────────────────────────────────────────
DEPLOYMENTS = [
    d.strip()
    for d in os.getenv("DEPLOYMENTS", "").split(",")
    if d.strip()
]
MAX_BUDGET_USD: float = float(os.getenv("MAX_BUDGET_USD", "10"))

MUNSIT_API_KEY: str = os.getenv("MUNSIT_API_KEY", "")
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
ROUTER_MODE: str = os.getenv("ROUTER_MODE", "difficulty")
MAX_STT_MINUTES: float = float(os.getenv("MAX_STT_MINUTES", "10"))

# Role env vars – raw values; resolved at startup by demo/deployments.py
STRUCTURING_DEPLOYMENT: str = os.getenv("STRUCTURING_DEPLOYMENT", "")
EXTRACTION_DEPLOYMENT: str = os.getenv("EXTRACTION_DEPLOYMENT", "")
TIER_SIMPLE: str = os.getenv("TIER_SIMPLE", "")
TIER_DEFAULT: str = os.getenv("TIER_DEFAULT", "")
TIER_ESCALATE: str = os.getenv("TIER_ESCALATE", "")

# ── Pricing ───────────────────────────────────────────────────────────────────
try:
    with open("prices.yaml", encoding="utf-8") as _f:
        prices: Dict = yaml.safe_load(_f).get("models", {})
except Exception:
    prices = {}

# ── Budget guard ─────────────────────────────────────────────────────────────
class BudgetManager:
    def __init__(self, max_budget: float) -> None:
        self.max_budget = max_budget
        self.current_spend: float = 0.0

    def add_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        mp = prices.get(model, {"input_1m": 0.0, "output_1m": 0.0})
        cost = (
            (input_tokens  / 1_000_000) * mp.get("input_1m", 0.0)
            + (output_tokens / 1_000_000) * mp.get("output_1m", 0.0)
        )
        self.current_spend += cost
        if self.current_spend > self.max_budget:
            raise RuntimeError(
                f"Budget exceeded: ${self.current_spend:.4f} > ${self.max_budget:.4f}"
            )
        return cost


budget_manager = BudgetManager(MAX_BUDGET_USD)

# ── Shared async client ───────────────────────────────────────────────────────
client = AsyncOpenAI(
    api_key=FOUNDRY_API_KEY,
    base_url=FOUNDRY_BASE_URL,
    max_retries=0,
    timeout=45.0,
)
