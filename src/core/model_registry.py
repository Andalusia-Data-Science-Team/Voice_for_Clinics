"""
Central registry of the OpenRouter models selectable per pipeline step.

Single source of truth for:
  * the /models endpoint (feeds the two dropdowns in the UI),
  * LLMService (resolves an id from the UI -> OpenRouter slug),
  * the pytest evaluation suite (which models to benchmark + which judge to use).

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
    id: str                          # stable id sent by the UI / used in logs
    label: str                       # what the dropdown shows
    slug: str                        # OpenRouter model slug
    supports_reasoning: bool = False  # accepts OpenRouter's `reasoning` param

    @property
    def provider(self) -> str:
        return self.slug.split("/", 1)[0].lower()


def _slug(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default).strip()


# ── Step 1: clinical note field extraction ─────────────────────────────────
EXTRACTION_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec("gpt-4o-mini",      "GPT 4o mini",      _slug("MODEL_SLUG_GPT_4O_MINI",   "openai/gpt-4o-mini")),
    ModelSpec("gpt-5.6-luna",     "GPT 5.6 Luna",     _slug("MODEL_SLUG_GPT_56_LUNA",   "openai/gpt-5.6-luna")),
    ModelSpec("glm-5.3-flash",    "GLM 5.3 Flash",    _slug("MODEL_SLUG_GLM_53_FLASH",  "z-ai/glm-5.3-flash")),
    ModelSpec("claude-sonnet-5.5", "Claude Sonnet 5.5", _slug("MODEL_SLUG_SONNET_55",   "anthropic/claude-sonnet-5.5")),
]}

# ── Step 2: ICD-10 diagnosis recommendation (uses reasoning) ───────────────
REASONING_MODELS: Dict[str, ModelSpec] = {m.id: m for m in [
    ModelSpec("gpt-4o-mini",       "GPT 4o mini",       _slug("MODEL_SLUG_GPT_4O_MINI",  "openai/gpt-4o-mini"), supports_reasoning=False),
    ModelSpec("gemini-3.7-flash",  "Gemini 3.7 Flash",  _slug("MODEL_SLUG_GEMINI_37_FLASH", "google/gemini-3.7-flash"), supports_reasoning=True),
    ModelSpec("claude-sonnet-5.5", "Claude Sonnet 5.5", _slug("MODEL_SLUG_SONNET_55",    "anthropic/claude-sonnet-5.5"), supports_reasoning=True),
    ModelSpec("claude-opus-5",     "Claude Opus 5",     _slug("MODEL_SLUG_OPUS_5",       "anthropic/claude-opus-5"), supports_reasoning=True),
]}

DEFAULT_EXTRACTION_MODEL = "gpt-4o-mini"
DEFAULT_REASONING_MODEL = "gpt-4o-mini"

# Reasoning effort sent to OpenRouter for reasoning-capable models in step 2.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")

# Output-token ceilings (reasoning tokens count toward max_tokens on OpenRouter,
# hence the much larger diagnosis budget).
EXTRACTION_MAX_TOKENS = int(os.environ.get("EXTRACTION_MAX_TOKENS", "4096"))
DIAGNOSIS_MAX_TOKENS = int(os.environ.get("DIAGNOSIS_MAX_TOKENS", "8000"))

# ── Judges for the evaluation suite ────────────────────────────────────────
# Anthropic models are judged by OpenAI; everything else by Claude.
JUDGE_FOR_ANTHROPIC = ModelSpec("gpt-5.6-sol", "GPT 5.6 Sol", _slug("JUDGE_SLUG_FOR_ANTHROPIC", "openai/gpt-5.6-sol"), True)
JUDGE_FOR_OTHERS = ModelSpec("claude-opus-5.5", "Claude Opus 5.5", _slug("JUDGE_SLUG_FOR_OTHERS", "anthropic/claude-opus-5.5"), True)


def judge_for(model: ModelSpec) -> ModelSpec:
    return JUDGE_FOR_ANTHROPIC if model.provider == "anthropic" else JUDGE_FOR_OTHERS


def get_model(step: str, model_id: Optional[str]) -> ModelSpec:
    """step: 'extraction' | 'reasoning'. Unknown/blank id -> ValueError
    (blank -> that step's default)."""
    registry = EXTRACTION_MODELS if step == "extraction" else REASONING_MODELS
    default = DEFAULT_EXTRACTION_MODEL if step == "extraction" else DEFAULT_REASONING_MODEL
    model_id = (model_id or "").strip() or default
    if model_id not in registry:
        raise ValueError(f"Unknown {step} model {model_id!r}. Available: {sorted(registry)}")
    return registry[model_id]


def registry_for_ui() -> dict:
    def dump(reg, default):
        return {"default": default, "models": [{"id": m.id, "label": m.label} for m in reg.values()]}
    return {
        "extraction": dump(EXTRACTION_MODELS, DEFAULT_EXTRACTION_MODEL),
        "reasoning": dump(REASONING_MODELS, DEFAULT_REASONING_MODEL),
    }