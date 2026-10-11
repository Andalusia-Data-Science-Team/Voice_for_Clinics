"""
demo/errors.py – Error helpers isolated from FastAPI so tests can import without
pulling in the full ASGI stack (avoids email-validator system version conflict).
Python 3.9 compatible. SECURITY: never leaks keys.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Keys to redact — read lazily at call time so tests can set env vars after import.
def redact(text: str) -> str:
    """Remove all API keys from a string."""
    for k in [
        os.getenv("FOUNDRY_API_KEY", ""),
        os.getenv("MUNSIT_API_KEY", ""),
        os.getenv("MISTRAL_API_KEY", ""),
    ]:
        if k and len(k) > 8:
            text = text.replace(k, "***REDACTED***")
    return text



def map_provider_error(
    exc: Exception,
    step: str,
    deployment: Optional[str],
) -> Dict[str, Any]:
    """
    Return a safe dict for a provider error — no keys, no prompts.
    Callers wrap this in a JSONResponse with the returned status_code.
    """
    import openai  # local import to avoid circular issues

    if isinstance(exc, openai.NotFoundError):
        status, msg = 502, "Deployment not found on the Azure resource."
    elif isinstance(exc, openai.AuthenticationError):
        status, msg = 502, "Authentication failed with the Azure endpoint."
    elif isinstance(exc, openai.PermissionDeniedError):
        status, msg = 502, "Permission denied by the Azure endpoint."
    elif isinstance(exc, openai.RateLimitError):
        status, msg = 429, "Rate limit reached. Retry after a moment."
    elif isinstance(exc, openai.APITimeoutError):
        status, msg = 504, "Request timed out."
    elif isinstance(exc, openai.APIConnectionError):
        status, msg = 502, "Cannot connect to Azure endpoint."
    elif isinstance(exc, RuntimeError) and "Budget" in str(exc):
        status, msg = 402, "Budget limit reached."
    else:
        status, msg = 502, f"Provider error: {type(exc).__name__}"

    logger.error(
        "Provider error [%s] deployment=%s: %s",
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
