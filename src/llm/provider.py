"""
llm/provider.py — Provider-aware kwargs builder and error mapper.

Two providers: "openrouter" (default) and "foundry".
Selected by LLM_PROVIDER env var at runtime init time.

SECURITY: never logs or exposes secrets, keys, prompts, or PHI.
Python 3.9 compatible (no X|Y unions, no match statements).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability profiles from foundry-probe/results/deployments_resolved.json
# ---------------------------------------------------------------------------
# Hard-coded for the models confirmed OK in the probe run.
# New deployments not listed here get a permissive profile (safe defaults).
# The profile is used ONLY when LLM_PROVIDER=foundry.
_FOUNDRY_PROFILES: Dict[str, Dict[str, bool]] = {
    "gpt-5.6-luna": {
        "max_completion_tokens": True,
        "max_tokens": False,
        "streaming": True,
        "json_object": False,
        "json_schema_strict": True,
        "temperature": False,
        "reasoning_effort": True,
    },
    "gpt-5.6-terra": {
        "max_completion_tokens": True,
        "max_tokens": False,
        "streaming": True,
        "json_object": False,
        "json_schema_strict": True,
        "temperature": False,
        "reasoning_effort": True,
    },
    "gpt-5.6-sol": {
        "max_completion_tokens": True,
        "max_tokens": False,
        "streaming": True,
        "json_object": False,
        "json_schema_strict": True,
        "temperature": False,
        "reasoning_effort": True,
    },
}

# Permissive fallback for unknown deployments (conservative defaults).
_PERMISSIVE_PROFILE: Dict[str, bool] = {
    "max_completion_tokens": True,
    "max_tokens": True,
    "streaming": True,
    "json_object": True,
    "json_schema_strict": False,
    "temperature": True,
    "reasoning_effort": False,
}


def get_foundry_profile(deployment_name: str) -> Dict[str, bool]:
    """Return the capability profile for a Foundry deployment.

    Falls back to a permissive profile for unknown deployments so new
    models don't silently break — callers can then discover issues from
    the provider's own error responses.
    """
    return _FOUNDRY_PROFILES.get(deployment_name, _PERMISSIVE_PROFILE)


def build_foundry_kwargs(
    deployment_name: str,
    *,
    messages: list,
    max_tokens: int,
    stream: bool = False,
    response_format: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a safe kwargs dict for a Foundry chat completion call.

    Omits parameters the deployment does not support (e.g. temperature,
    json_object, max_tokens legacy alias).  Falls back from json_schema_strict
    to json_object to no response_format, in that order.

    Never includes OpenRouter-specific keys: extra_body usage.include,
    extra_body reasoning.
    """
    profile = get_foundry_profile(deployment_name)
    kwargs: Dict[str, Any] = {
        "model": deployment_name,
        "messages": messages,
    }

    # Token limit parameter — max_completion_tokens preferred
    if profile.get("max_completion_tokens", True):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    # Temperature — only if the model accepts it (Luna does not)
    # For Foundry we never send temperature=0 to reasoning models.
    # Omit entirely when not supported.

    # Streaming
    if stream and profile.get("streaming", True):
        kwargs["stream"] = True

    # Response format
    if response_format:
        rf_type = response_format.get("type", "")
        if rf_type == "json_schema":
            if profile.get("json_schema_strict", False):
                kwargs["response_format"] = response_format
            elif profile.get("json_object", True):
                kwargs["response_format"] = {"type": "json_object"}
                logger.debug(
                    "[Foundry] %s: json_schema_strict unsupported, downgraded to json_object",
                    deployment_name,
                )
            # else: omit response_format entirely — model free-forms
        elif rf_type == "json_object":
            if profile.get("json_object", True):
                kwargs["response_format"] = response_format
            # else: omit
        else:
            kwargs["response_format"] = response_format

    # Reasoning effort (Foundry-native param for o-series / Luna-class models)
    if reasoning_effort and profile.get("reasoning_effort", False):
        kwargs["reasoning_effort"] = reasoning_effort

    return kwargs


# ---------------------------------------------------------------------------
# Provider error mapping (no secrets in messages)
# Ported from foundry-probe/demo/errors.py
# ---------------------------------------------------------------------------

def map_provider_error(exc: Exception, step: str, deployment: Optional[str]) -> Dict[str, Any]:
    """Return a safe dict describing a provider error. No keys or PHI.

    Callers should log exc type (not message) and return the body to clients.
    """
    import openai  # local import — avoid circular at module load

    if isinstance(exc, openai.NotFoundError):
        status, msg = 502, "Deployment not found on the Azure resource."
    elif isinstance(exc, openai.AuthenticationError):
        status, msg = 502, "Authentication failed with the provider endpoint."
    elif isinstance(exc, openai.PermissionDeniedError):
        status, msg = 502, "Permission denied by the provider endpoint."
    elif isinstance(exc, openai.RateLimitError):
        status, msg = 429, "Rate limit reached. Retry after a moment."
    elif isinstance(exc, openai.APITimeoutError):
        status, msg = 504, "Request timed out."
    elif isinstance(exc, openai.APIConnectionError):
        status, msg = 502, "Cannot connect to provider endpoint."
    else:
        status, msg = 502, f"Provider error: {type(exc).__name__}"

    logger.error(
        "[LLM] Provider error [%s] deployment=%s: %s",
        step, deployment, type(exc).__name__,
    )
    return {
        "status_code": status,
        "body": {
            "error": type(exc).__name__,
            "message": msg,
            "step": step,
            "deployment": deployment,
        },
    }
