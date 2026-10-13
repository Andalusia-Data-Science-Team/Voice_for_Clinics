"""
Central registry of Azure AI Foundry models selectable per pipeline step.

Single source of truth for:
  * the /models endpoint (feeds the two dropdowns in the UI),
  * LLMService (resolves model id -> Foundry deployment name),
  * the evaluation and diagnostic suites.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    id: str                          # stable id sent by the UI / used in logs
    label: str                       # what the dropdown shows
    foundry_deployment: str          # Azure AI Foundry deployment name
    supports_reasoning: bool = False  # accepts reasoning_effort param

    @property
    def provider(self) -> str:
        return "foundry"

    @property
    def slug(self) -> str:
        return self.foundry_deployment


def _dep(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default).strip()


# ── Step 1: clinical note field extraction ─────────────────────────────────
EXTRACTION_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec(
        "gpt-5.6-luna", "GPT 5.6 Luna",
        _dep("FOUNDRY_DEPLOYMENT_LUNA", "gpt-5.6-luna"),
        supports_reasoning=True,
    ),
    ModelSpec(
        "gpt-5.6-terra", "GPT 5.6 Terra",
        _dep("FOUNDRY_DEPLOYMENT_TERRA", "gpt-5.6-terra"),
        supports_reasoning=True,
    ),
    ModelSpec(
        "gpt-5.6-sol", "GPT 5.6 Sol",
        _dep("FOUNDRY_DEPLOYMENT_SOL", "gpt-5.6-sol"),
        supports_reasoning=True,
    ),
    ModelSpec(
        "DeepSeek-V3.2", "DeepSeek V3.2",
        _dep("FOUNDRY_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.2"),
        supports_reasoning=False,
    ),
]}

# ── Step 2: ICD-10 diagnosis recommendation (uses reasoning) ───────────────
REASONING_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec(
        "gpt-5.6-luna", "GPT 5.6 Luna",
        _dep("FOUNDRY_DEPLOYMENT_LUNA", "gpt-5.6-luna"),
        supports_reasoning=True,
    ),
    ModelSpec(
        "gpt-5.6-terra", "GPT 5.6 Terra",
        _dep("FOUNDRY_DEPLOYMENT_TERRA", "gpt-5.6-terra"),
        supports_reasoning=True,
    ),
    ModelSpec(
        "gpt-5.6-sol", "GPT 5.6 Sol",
        _dep("FOUNDRY_DEPLOYMENT_SOL", "gpt-5.6-sol"),
        supports_reasoning=True,
    ),
    ModelSpec(
        "DeepSeek-V3.2", "DeepSeek V3.2",
        _dep("FOUNDRY_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.2"),
        supports_reasoning=False,
    ),
]}

DEFAULT_EXTRACTION_MODEL = os.environ.get(
    "FOUNDRY_DEFAULT_EXTRACTION_MODEL", "gpt-5.6-luna"
).strip()
DEFAULT_REASONING_MODEL = os.environ.get(
    "FOUNDRY_DEFAULT_REASONING_MODEL", "gpt-5.6-luna"
).strip()

FOUNDRY_DEFAULT_EXTRACTION_MODEL = DEFAULT_EXTRACTION_MODEL
FOUNDRY_DEFAULT_REASONING_MODEL = DEFAULT_REASONING_MODEL

# Reasoning effort sent to Foundry for reasoning-capable models in step 2.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")

# Output-token ceilings
EXTRACTION_MAX_TOKENS = int(os.environ.get("EXTRACTION_MAX_TOKENS", "4096"))
DIAGNOSIS_MAX_TOKENS = int(os.environ.get("DIAGNOSIS_MAX_TOKENS", "8000"))

# ── Judges for evaluation ──────────────────────────────────────────────────
JUDGE_FOR_OPENAI = ModelSpec(
    "gpt-5.6-sol",
    "GPT 5.6 Sol",
    _dep("FOUNDRY_DEPLOYMENT_SOL", "gpt-5.6-sol"),
    True,
)
JUDGE_FOR_NON_OPENAI = JUDGE_FOR_OPENAI


def judge_for(model: ModelSpec) -> ModelSpec:
    """Select the judge model for evaluation."""
    return JUDGE_FOR_OPENAI


def get_model(step: str, model_id: Optional[str]) -> ModelSpec:
    """step: 'extraction' | 'reasoning'. Unknown/blank id -> ValueError."""
    registry = EXTRACTION_MODELS if step == "extraction" else REASONING_MODELS
    default = DEFAULT_EXTRACTION_MODEL if step == "extraction" else DEFAULT_REASONING_MODEL

    model_id = (model_id or "").strip() or default
    if model_id not in registry:
        for m in registry.values():
            if m.foundry_deployment.lower() == model_id.lower():
                return m
        raise ValueError(
            f"Unknown {step} model {model_id!r}. Available: {sorted(registry)}"
        )
    return registry[model_id]


def registry_for_ui() -> dict:
    """Return the models shown by GET /models."""
    def dump(reg: Dict[str, ModelSpec], default: str) -> dict:
        models = [{"id": m.id, "label": m.label} for m in reg.values()]
        return {
            "default": default if default in reg else next(iter(reg)),
            "models": models,
        }

    return {
        "extraction": dump(EXTRACTION_MODELS, DEFAULT_EXTRACTION_MODEL),
        "reasoning": dump(REASONING_MODELS, DEFAULT_REASONING_MODEL),
    }