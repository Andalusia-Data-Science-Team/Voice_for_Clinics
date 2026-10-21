#!/usr/bin/env python3
"""A2.5.2: verify Foundry accepts Entra ID tokens and rejects API keys.

CREATE ONLY — the owner runs this after disabling local (API-key) auth in the
Azure portal. It makes live calls to Azure AI Foundry.

Expected outcome once API-key auth is disabled:
  * an API-key call is REJECTED (401/403);
  * a token (managed identity / Entra ID) call SUCCEEDS.

No secrets are printed. Only pass/fail and status codes are shown.
"""

from __future__ import annotations

import asyncio
import os
import sys


def _base_url() -> str:
    endpoint = os.getenv("AI_FOUNDRY_ENDPOINT") or os.getenv("FOUNDRY_BASE_URL") or ""
    base = endpoint.rstrip("/")
    if base and not base.endswith("/openai/v1"):
        base = f"{base}/openai/v1"
    return base


async def _try_api_key(base_url: str, deployment: str) -> int | None:
    """Return the HTTP status of an API-key call, or None on transport error."""
    from openai import AsyncOpenAI

    api_key = os.getenv("AI_FOUNDRY_KEY") or os.getenv("FOUNDRY_API_KEY") or os.getenv("AZURE_API_KEY")
    if not api_key:
        return None
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=20.0)
    try:
        await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return 200
    except Exception as exc:
        return getattr(exc, "status_code", None)
    finally:
        await client.close()


async def _try_token(base_url: str, deployment: str) -> int | None:
    """Return the HTTP status of a managed-identity token call, or None."""
    from openai import AsyncOpenAI

    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except Exception:
        return None
    credential = DefaultAzureCredential(logging_enable=False)
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client = AsyncOpenAI(
        base_url=base_url,
        azure_ad_token_provider=token_provider,
        max_retries=0,
        timeout=20.0,
    )
    try:
        await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return 200
    except Exception as exc:
        return getattr(exc, "status_code", None)
    finally:
        await client.close()


async def main() -> int:
    base_url = _base_url()
    if not base_url:
        print("AI_FOUNDRY_ENDPOINT is not set; cannot run.", file=sys.stderr)
        return 2
    deployment = os.getenv("FOUNDRY_MODEL_EXTRACTION", "gpt-5.6-luna")

    api_status = await _try_api_key(base_url, deployment)
    token_status = await _try_token(base_url, deployment)

    api_rejected = api_status in (401, 403)
    token_ok = token_status == 200

    print(f"api_key_call_status={api_status} (expected 401/403)")
    print(f"token_call_status={token_status} (expected 200)")
    print(f"API_KEY_REJECTED={'PASS' if api_rejected else 'FAIL'}")
    print(f"TOKEN_ACCEPTED={'PASS' if token_ok else 'FAIL'}")
    overall = api_rejected and token_ok
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))