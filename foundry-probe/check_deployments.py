"""
check_deployments.py – Standalone discovery script.
Usage: python check_deployments.py
Writes results/deployments_resolved.json.
Python 3.9 compatible.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Must be set before importing demo modules
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY", "")
FOUNDRY_BASE_URL = os.getenv("FOUNDRY_BASE_URL", "")

if not FOUNDRY_API_KEY or not FOUNDRY_BASE_URL:
    print("ERROR: FOUNDRY_API_KEY and FOUNDRY_BASE_URL must be set in .env")
    sys.exit(1)

from openai import AsyncOpenAI
from demo.deployments import discover, resolve_roles

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def main() -> None:
    client = AsyncOpenAI(
        api_key=FOUNDRY_API_KEY,
        base_url=FOUNDRY_BASE_URL,
        max_retries=0,
        timeout=25.0,
    )

    # Extra candidates from env
    extra: list = []
    for var in ("DEPLOYMENTS", "STRUCTURING_DEPLOYMENT", "EXTRACTION_DEPLOYMENT",
                "TIER_SIMPLE", "TIER_DEFAULT", "TIER_ESCALATE"):
        val = os.getenv(var, "")
        if "," in val:
            extra.extend([v.strip() for v in val.split(",") if v.strip()])
        elif val:
            extra.append(val)

    result = await discover(client, FOUNDRY_API_KEY, extra_names=extra)
    roles = resolve_roles(result["ok"])

    print("\n── Resolved Role Map ─────────────────────────────────────────")
    for role, dep in roles.items():
        status = "✓" if dep else "✗ UNAVAILABLE"
        print(f"  {role:<20} → {dep or '(none)'} {status}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
