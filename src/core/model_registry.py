"""
Central registry of LLM models used per pipeline step.

Single source of truth for:
  * the /models endpoint (reports the fixed pipeline models),
  * LLMService (resolves a model id -> OpenRouter slug or Foundry deployment),
  * the pytest evaluation suite (which models to benchmark + which judge to use).

Provider selection: LLM_PROVIDER=openrouter (default) | foundry
  - openrouter: uses ModelSpec.slug (e.g. "openai/gpt-5.6-luna")
  - foundry:    uses ModelSpec.foundry_deployment (e.g. "gpt-5.6-luna")
                GET /models returns only models with a Foundry deployment.
                Defaults: extraction=gpt-5.6-luna, reasoning=gpt-5.6-luna.

!! VERIFY THE SLUGS !!  Model names move fast. Every slug below can be
overridden without touching code via an env var (see `_slug`), and the test
suite cross-checks them against https://openrouter.ai/api/v1/models and skips
(with a clear message) any slug OpenRouter doesn't list.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    id: str                           # stable id used in logs and API responses
    label: str                        # display label for model metadata
    slug: str                         # OpenRouter model slug
    supports_reasoning: bool = False  # accepts OpenRouter's `reasoning` param
    foundry_deployment: Optional[str] = None  # Azure AI Foundry deployment name, or None

    @property
    def provider(self) -> str:
        return self.slug.split("/", 1)[0].lower()


def _slug(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default).strip()


def _foundry_dep(env_name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a Foundry deployment name from env, or the hard-coded default."""
    val = os.environ.get(env_name, "").strip()
    return val if val else default


# ── Step 1: clinical note field extraction ─────────────────────────────────
EXTRACTION_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec(
        "gpt-4o-mini",
        "GPT 4o mini",
        _slug("MODEL_SLUG_GPT_4O_MINI", "openai/gpt-4o-mini"),
        foundry_deployment=None,          # NOT available on Foundry (confirmed not_found)
    ),
    ModelSpec(
        "gpt-5.6-luna",
        "GPT 5.6 Luna",
        _slug("MODEL_SLUG_GPT_56_LUNA", "openai/gpt-5.6-luna"),
        foundry_deployment=_foundry_dep("FOUNDRY_DEPLOYMENT_LUNA", "gpt-5.6-luna"),
    ),
    ModelSpec(
        "glm-5.3-flash",
        "GLM 5.3 Flash",
        _slug("MODEL_SLUG_GLM_53_FLASH", "z-ai/glm-5.3-flash"),
        foundry_deployment=None,          # not on Foundry
    ),
    ModelSpec(
        "claude-sonnet-5.5",
        "Claude Sonnet 5.5",
        _slug("MODEL_SLUG_SONNET_55", "anthropic/claude-sonnet-5.5"),
        foundry_deployment=None,          # not on Foundry
    ),
]}

# ── Step 2: ICD-10 diagnosis recommendation (uses reasoning) ───────────────
REASONING_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec(
        "gpt-4o-mini",
        "GPT 4o mini",
        _slug("MODEL_SLUG_GPT_4O_MINI", "openai/gpt-4o-mini"),
        supports_reasoning=False,
        foundry_deployment=None,          # NOT available on Foundry
    ),
    ModelSpec(
        "gpt-5.6-luna",
        "GPT 5.6 Luna",
        _slug("MODEL_SLUG_GPT_56_LUNA", "openai/gpt-5.6-luna"),
        supports_reasoning=False,
        foundry_deployment=_foundry_dep("FOUNDRY_DEPLOYMENT_LUNA", "gpt-5.6-luna"),
    ),
    ModelSpec(
        "gemini-3.7-flash",
        "Gemini 3.7 Flash",
        _slug("MODEL_SLUG_GEMINI_37_FLASH", "google/gemini-3.7-flash"),
        supports_reasoning=True,
        foundry_deployment=None,
    ),
    ModelSpec(
        "claude-sonnet-5.5",
        "Claude Sonnet 5.5",
        _slug("MODEL_SLUG_SONNET_55", "anthropic/claude-sonnet-5.5"),
        supports_reasoning=True,
        foundry_deployment=None,
    ),
    ModelSpec(
        "claude-opus-5",
        "Claude Opus 5",
        _slug("MODEL_SLUG_OPUS_5", "anthropic/claude-opus-5"),
        supports_reasoning=True,
        foundry_deployment=None,
    ),
]}

# OpenRouter defaults (unchanged)
DEFAULT_EXTRACTION_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_MODEL = "gpt-5.6-luna"

# Foundry defaults — both steps use Luna; configurable via env without code change
FOUNDRY_DEFAULT_EXTRACTION_MODEL = os.environ.get(
    "FOUNDRY_DEFAULT_EXTRACTION_MODEL", "gpt-5.6-luna"
).strip()
FOUNDRY_DEFAULT_REASONING_MODEL = os.environ.get(
    "FOUNDRY_DEFAULT_REASONING_MODEL", "gpt-5.6-luna"
).strip()

# Reasoning effort sent to OpenRouter for reasoning-capable models in step 2.
# Not sent to Foundry (Foundry uses reasoning_effort at the deployment level).
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")

# Output-token ceilings (reasoning tokens count toward max_tokens on OpenRouter,
# hence the much larger diagnosis budget).
EXTRACTION_MAX_TOKENS = int(os.environ.get("EXTRACTION_MAX_TOKENS", "4096"))
DIAGNOSIS_MAX_TOKENS = int(os.environ.get("DIAGNOSIS_MAX_TOKENS", "8000"))

# ── Judges for the evaluation suite ────────────────────────────────────────
# Live eval suite stays on OpenRouter regardless of LLM_PROVIDER.
JUDGE_FOR_ANTHROPIC = ModelSpec(
    "gpt-5.6-sol", "GPT 5.6 Sol",
    _slug("JUDGE_SLUG_FOR_ANTHROPIC", "openai/gpt-5.6-sol"), True
)
JUDGE_FOR_OTHERS = ModelSpec(
    "claude-opus-5.5", "Claude Opus 5.5",
    _slug("JUDGE_SLUG_FOR_OTHERS", "anthropic/claude-opus-5.5"), True
)


def judge_for(model: ModelSpec) -> ModelSpec:
    return JUDGE_FOR_ANTHROPIC if model.provider == "anthropic" else JUDGE_FOR_OTHERS


def _active_provider() -> str:
    """Return 'openrouter' or 'foundry' based on current env."""
    return os.environ.get("LLM_PROVIDER", "openrouter").lower()


def get_model(step: str, model_id: Optional[str]) -> ModelSpec:
    """step: 'extraction' | 'reasoning'. Unknown/blank id -> ValueError
    (blank -> that step's default, which is provider-aware)."""
    provider = _active_provider()
    registry = EXTRACTION_MODELS if step == "extraction" else REASONING_MODELS

    if provider == "foundry":
        # Foundry: use provider-specific defaults
        default = (
            FOUNDRY_DEFAULT_EXTRACTION_MODEL if step == "extraction"
            else FOUNDRY_DEFAULT_REASONING_MODEL
        )
    else:
        # OpenRouter: use original defaults
        default = DEFAULT_EXTRACTION_MODEL if step == "extraction" else DEFAULT_REASONING_MODEL

    model_id = (model_id or "").strip() or default

    if model_id not in registry:
        raise ValueError(
            f"Unknown {step} model {model_id!r}. Available: {sorted(registry)}"
        )

    spec = registry[model_id]

    # Foundry: reject models that have no Foundry deployment configured
    if provider == "foundry" and spec.foundry_deployment is None:
        raise ValueError(
            f"Model {model_id!r} has no Foundry deployment configured. "
            f"Set FOUNDRY_DEPLOYMENT_* env vars or choose a model with a deployment."
        )

    return spec


def registry_for_ui() -> dict:
    """Return the models shown by GET /models.

    With provider=foundry, only models that have a Foundry deployment are returned.
    With provider=openrouter, behavior is unchanged from before (defaults only).
    """
    provider = _active_provider()

    def dump(reg: Dict[str, ModelSpec], default: str) -> dict:
        if provider == "foundry":
            available = [m for m in reg.values() if m.foundry_deployment is not None]
            if not available:
                return {"default": default, "models": []}
            # Use the Foundry default if it's in the available set, else first available
            foundry_default_id = (
                FOUNDRY_DEFAULT_EXTRACTION_MODEL
                if reg is EXTRACTION_MODELS
                else FOUNDRY_DEFAULT_REASONING_MODEL
            )
            actual_default = (
                foundry_default_id
                if any(m.id == foundry_default_id for m in available)
                else available[0].id
            )
            return {
                "default": actual_default,
                "models": [{"id": m.id, "label": m.label} for m in available],
            }
        else:
            model = reg[default]
            return {"default": default, "models": [{"id": model.id, "label": model.label}]}

    return {
        "extraction": dump(EXTRACTION_MODELS, DEFAULT_EXTRACTION_MODEL),
        "reasoning": dump(REASONING_MODELS, DEFAULT_REASONING_MODEL),
    }