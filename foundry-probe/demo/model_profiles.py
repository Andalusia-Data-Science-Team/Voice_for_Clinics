"""
demo/model_profiles.py – Per-deployment parameter profiles.
Loaded from results/deployments_resolved.json (written by discovery).
Python 3.9 compatible.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-memory profile store: name -> supports dict
_PROFILES: Dict[str, Dict[str, bool]] = {}


def load_from_resolved(path: str = "results/deployments_resolved.json") -> None:
    """Load profiles from the resolved deployments file."""
    global _PROFILES
    if not os.path.exists(path):
        logger.warning("deployments_resolved.json not found; profiles empty")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for dep in data.get("ok", []):
        _PROFILES[dep["name"]] = dep.get("supports", {})
    logger.info("Loaded %d model profiles from %s", len(_PROFILES), path)


def get_profile(name: str) -> Dict[str, bool]:
    """Return the capability profile for a deployment. Unknown → permissive."""
    return _PROFILES.get(name, {
        "max_completion_tokens": True,
        "max_tokens": True,
        "streaming": True,
        "json_object": True,
        "json_schema_strict": False,
        "temperature": True,
        "reasoning_effort": False,
    })


def build_kwargs(
    name: str,
    *,
    max_tokens: int,
    temperature: Optional[float] = None,
    stream: bool = False,
    response_format: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a safe kwargs dict for client.chat.completions.create,
    dropping parameters the deployment doesn't support.
    Falls back from json_schema_strict → json_object.
    Falls back from max_completion_tokens → max_tokens.
    """
    profile = get_profile(name)
    kwargs: Dict[str, Any] = {}

    # Token limit parameter
    if profile.get("max_completion_tokens", True):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    # Temperature
    if temperature is not None and profile.get("temperature", True):
        kwargs["temperature"] = temperature

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
                # Downgrade: caller must validate locally
                kwargs["response_format"] = {"type": "json_object"}
                logger.debug(
                    "%s: json_schema_strict unsupported, fell back to json_object", name
                )
            # else: no response_format set, model free-forms
        elif rf_type == "json_object":
            if profile.get("json_object", True):
                kwargs["response_format"] = response_format
        else:
            kwargs["response_format"] = response_format

    # Reasoning effort
    if reasoning_effort and profile.get("reasoning_effort", False):
        kwargs["reasoning_effort"] = reasoning_effort

    return kwargs
