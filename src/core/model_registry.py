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
    selectable: bool = True          # False for fallback-only / router-only deployments

    @property
    def provider(self) -> str:
        return "foundry"

    @property
    def slug(self) -> str:
        return self.foundry_deployment


def _dep(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default).strip()


# ── Step 1: clinical note field extraction ─────────────────────────────────
# Primary: gpt-5.6-luna. Fallback: gpt-4.1-mini (fallback-only, not selectable).
# Terra/Sol/DeepSeek are not valid extraction primaries.
EXTRACTION_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec(
        "gpt-5.6-luna", "GPT 5.6 Luna",
        _dep("FOUNDRY_DEPLOYMENT_LUNA", "gpt-5.6-luna"),
        supports_reasoning=True,
        selectable=True,
    ),
    ModelSpec(
        "gpt-5.6-terra", "GPT 5.6 Terra",
        _dep("FOUNDRY_DEPLOYMENT_TERRA", "gpt-5.6-terra"),
        supports_reasoning=True,
        selectable=False,
    ),
    ModelSpec(
        "gpt-5.6-sol", "GPT 5.6 Sol",
        _dep("FOUNDRY_DEPLOYMENT_SOL", "gpt-5.6-sol"),
        supports_reasoning=True,
        selectable=False,
    ),
    ModelSpec(
        "gpt-4.1-mini", "GPT 4.1 Mini",
        _dep("FOUNDRY_DEPLOYMENT_GPT_4_1_MINI", "gpt-4.1-mini"),
        supports_reasoning=False,
        selectable=False,
    ),
    ModelSpec(
        "DeepSeek-V3.2", "DeepSeek V3.2",
        _dep("FOUNDRY_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.2"),
        supports_reasoning=False,
        selectable=False,
    ),
]}

# ── Step 2: ICD-10 diagnosis recommendation (uses reasoning) ───────────────
# Primaries: gpt-5.6-luna (simple tier), gpt-5.6-sol (complex tier).
# Fallbacks: gpt-4.1-mini (simple), gpt-5.6-terra (complex) — not selectable.
REASONING_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec(
        "gpt-5.6-luna", "GPT 5.6 Luna",
        _dep("FOUNDRY_DEPLOYMENT_LUNA", "gpt-5.6-luna"),
        supports_reasoning=True,
        selectable=True,
    ),
    ModelSpec(
        "gpt-5.6-sol", "GPT 5.6 Sol",
        _dep("FOUNDRY_DEPLOYMENT_SOL", "gpt-5.6-sol"),
        supports_reasoning=True,
        selectable=True,
    ),
    ModelSpec(
        "gpt-5.6-terra", "GPT 5.6 Terra",
        _dep("FOUNDRY_DEPLOYMENT_TERRA", "gpt-5.6-terra"),
        supports_reasoning=True,
        selectable=False,
    ),
    ModelSpec(
        "gpt-4.1-mini", "GPT 4.1 Mini",
        _dep("FOUNDRY_DEPLOYMENT_GPT_4_1_MINI", "gpt-4.1-mini"),
        supports_reasoning=False,
        selectable=False,
    ),
    ModelSpec(
        "DeepSeek-V3.2", "DeepSeek V3.2",
        _dep("FOUNDRY_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.2"),
        supports_reasoning=False,
        selectable=False,
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


def get_model(step: str, model_id: Optional[str] = None, escalation: bool = False) -> ModelSpec:
    """step: 'extraction' | 'reasoning'. Unknown/blank id -> ValueError.
    If escalation=True for reasoning, returns the higher-tier Sol model (gpt-5.6-sol).

    An explicitly requested model that is fallback-only or router-only
    (``selectable=False``) is rejected, so a doctor cannot select a deployment
    that is not a valid primary.
    """
    registry = EXTRACTION_MODELS if step == "extraction" else REASONING_MODELS
    default = DEFAULT_EXTRACTION_MODEL if step == "extraction" else DEFAULT_REASONING_MODEL

    if escalation and step == "reasoning":
        # Sol model is the designated escalation model for reasoning failures
        if "gpt-5.6-sol" in registry:
            return registry["gpt-5.6-sol"]

    requested = (model_id or "").strip()
    if not requested:
        return registry[default]

    if requested in registry:
        spec = registry[requested]
    else:
        spec = next(
            (m for m in registry.values() if m.foundry_deployment.lower() == requested.lower()),
            None,
        )
        if spec is None:
            raise ValueError(
                f"Unknown {step} model {requested!r}. Available: {sorted(registry)}"
            )
    if not spec.selectable:
        raise ValueError(
            f"{step} model {requested!r} is not a selectable primary "
            f"(fallback-only or router-only deployment)."
        )
    return spec


def registry_for_ui() -> dict:
    """Return the models shown by GET /models.

    Only *selectable* primaries are listed. Fallback-only and router-only
    deployments (gpt-4.1-mini, gpt-5.6-terra, DeepSeek-V3.2, model-router) are
    never offered as doctor-selectable primaries.
    """
    def dump(reg: Dict[str, ModelSpec], default: str) -> dict:
        selectable = [m for m in reg.values() if m.selectable]
        models = [{"id": m.id, "label": m.label} for m in selectable]
        default_id = default if any(m.id == default for m in selectable) else (
            selectable[0].id if selectable else default
        )
        return {
            "default": default_id,
            "models": models,
        }

    return {
        "extraction": dump(EXTRACTION_MODELS, DEFAULT_EXTRACTION_MODEL),
        "reasoning": dump(REASONING_MODELS, DEFAULT_REASONING_MODEL),
    }


def get_model_by_deployment(deployment: str) -> Optional[ModelSpec]:
    """Resolve a Foundry deployment name to a ModelSpec, or None if unknown.

    Searches both registries; used by the routing/fallback layer to label the
    deployment that actually served a call.
    """
    target = (deployment or "").strip().lower()
    if not target:
        return None
    for reg in (EXTRACTION_MODELS, REASONING_MODELS):
        for spec in reg.values():
            if spec.foundry_deployment.lower() == target:
                return spec
    return None


def spec_for_deployment(deployment: str, *, supports_reasoning: bool = False) -> ModelSpec:
    """Return a ModelSpec for an arbitrary deployment name.

    Falls back to a synthetic spec when the deployment is not in a registry
    (e.g. the router deployment), so callers can still label and call it.
    """
    found = get_model_by_deployment(deployment)
    if found is not None:
        return found
    return ModelSpec(
        id=deployment,
        label=deployment,
        foundry_deployment=deployment,
        supports_reasoning=supports_reasoning,
        selectable=False,
    )