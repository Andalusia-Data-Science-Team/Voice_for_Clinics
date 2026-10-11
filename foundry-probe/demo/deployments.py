"""
demo/deployments.py – Deployment discovery, role resolution, capability probing.
Python 3.9 compatible (no X|Y unions, no match, no asyncio.TaskGroup).
SECURITY: never logs keys, prompts, or patient data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Any

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ── candidate list ────────────────────────────────────────────────────────────
BUILTIN_CANDIDATES: List[str] = [
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
    "gpt-6-luna", "gpt-6-sol",
    "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5.1", "gpt-5-mini",
    "DeepSeek-V3.2", "DeepSeek-V3", "DeepSeek-V4-Flash", "DeepSeek-V4-Pro",
]

# ── role preference by family keyword ─────────────────────────────────────────
ROLE_PREFS: Dict[str, List[str]] = {
    "STRUCTURING":  ["luna", "deepseek", "4.1-mini", "4o-mini"],
    "EXTRACTION":   ["4o-mini", "4.1-mini", "luna"],
    "TIER_SIMPLE":  ["luna", "4.1-mini", "4o-mini"],
    "TIER_DEFAULT": ["terra", "4.1", "deepseek-v4", "deepseek-v3"],
    "TIER_ESCALATE":["sol",  "gpt-5", "deepseek-v4-pro"],
}

# Cache so discovery doesn't run every request
_CACHE: Optional[Dict[str, Any]] = None
_CACHE_TS: float = 0.0
CACHE_TTL = 600  # 10 minutes


def _redact_key(text: str, key: str) -> str:
    if key and len(key) > 8:
        return text.replace(key, "***REDACTED***")
    return text


def _family(name: str) -> str:
    n = name.lower()
    if "luna"  in n: return "luna"
    if "terra" in n: return "terra"
    if "sol"   in n: return "sol"
    if "deepseek-v4-pro" in n: return "deepseek-v4-pro"
    if "deepseek-v4" in n: return "deepseek-v4"
    if "deepseek" in n: return "deepseek"
    if "4o-mini" in n or "4.1-mini" in n or "5-mini" in n: return "mini"
    if "4.1" in n: return "gpt-4.1"
    if "5.1" in n: return "gpt-5.1"
    return "unknown"


async def _probe_one(
    client: AsyncOpenAI, name: str, api_key: str
) -> Dict[str, Any]:
    """Probe a single deployment. Returns capability dict."""
    result: Dict[str, Any] = {
        "name": name, "status": "unknown", "latency_ms": None,
        "family": _family(name), "supports": {}
    }
    tiny_prompt = [{"role": "user", "content": "Hi"}]

    async def _try(**kwargs: Any) -> Any:
        return await asyncio.wait_for(
            client.chat.completions.create(
                model=name, messages=tiny_prompt, **kwargs
            ),
            timeout=20.0,
        )

    # --- primary probe with max_completion_tokens ---
    t0 = time.perf_counter()
    try:
        resp = await _try(max_completion_tokens=16)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        result["status"] = "ok"
        result["supports"]["max_completion_tokens"] = True
        result["supports"]["max_tokens"] = False
    except openai.NotFoundError:
        result["status"] = "not_found"
        return result
    except openai.AuthenticationError:
        result["status"] = "auth_error"
        return result
    except openai.PermissionDeniedError:
        result["status"] = "permission_denied"
        return result
    except openai.RateLimitError:
        result["status"] = "rate_limited"
        return result
    except (openai.BadRequestError, openai.UnprocessableEntityError):
        # max_completion_tokens not supported – retry with max_tokens
        try:
            t0 = time.perf_counter()
            resp = await _try(max_tokens=16)
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
            result["status"] = "ok"
            result["supports"]["max_tokens"] = True
            result["supports"]["max_completion_tokens"] = False
        except Exception as e2:
            result["status"] = f"error:{type(e2).__name__}"
            logger.warning("probe %s failed: %s", name, type(e2).__name__)
            return result
    except asyncio.TimeoutError:
        result["status"] = "timeout"
        return result
    except Exception as exc:
        result["status"] = f"error:{type(exc).__name__}"
        logger.warning("probe %s error: %s", name, type(exc).__name__)
        return result

    if result["status"] != "ok":
        return result

    # --- capability sub-probes (best-effort, don't crash) ---
    cap_key = (
        "max_completion_tokens"
        if result["supports"].get("max_completion_tokens")
        else "max_tokens"
    )

    async def _cap(feature: str, **kwargs: Any) -> bool:
        try:
            await asyncio.wait_for(
                client.chat.completions.create(
                    model=name, messages=tiny_prompt,
                    **{cap_key: 8}, **kwargs
                ),
                timeout=15.0,
            )
            return True
        except (openai.BadRequestError, openai.UnprocessableEntityError,
                openai.NotFoundError):
            return False
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    result["supports"]["streaming"] = await _cap(
        "streaming", stream=True
    )
    result["supports"]["json_object"] = await _cap(
        "json_object",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": "Return JSON."},
                  {"role": "user", "content": "{}"}],
    )
    result["supports"]["json_schema_strict"] = await _cap(
        "json_schema_strict",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "t", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"v": {"type": "string"}},
                    "required": ["v"],
                    "additionalProperties": False,
                },
            },
        },
    )
    result["supports"]["temperature"] = await _cap(
        "temperature", temperature=0.0
    )
    result["supports"]["reasoning_effort"] = await _cap(
        "reasoning_effort", reasoning_effort="low"
    )

    return result


async def discover(
    client: AsyncOpenAI,
    api_key: str,
    extra_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run full discovery. Returns structured results dict.
    Caches for CACHE_TTL seconds.
    """
    global _CACHE, _CACHE_TS
    if _CACHE and (time.time() - _CACHE_TS) < CACHE_TTL:
        return _CACHE

    # Gather candidate names
    candidates: List[str] = list(BUILTIN_CANDIDATES)
    if extra_names:
        for n in extra_names:
            if n and n not in candidates:
                candidates.append(n)

    # Try models.list() for additional names
    try:
        models_resp = await asyncio.wait_for(client.models.list(), timeout=10.0)
        listed = [m.id for m in models_resp.data]
        logger.info("models.list() returned %d entries", len(listed))
        # Only add names that look like our target families to avoid probing 300 models
        patterns = re.compile(
            r"gpt-5\.6|gpt-6|gpt-4\.1|gpt-5\.|4o-mini|deepseek|luna|terra|sol",
            re.IGNORECASE,
        )
        for m_id in listed:
            if patterns.search(m_id) and m_id not in candidates:
                candidates.append(m_id)
    except Exception as exc:
        logger.info("models.list() skipped: %s", type(exc).__name__)

    # Deduplicate preserving order, also add lowercase variants
    seen: set = set()
    unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
        lc = c.lower()
        if lc != c and lc not in seen:
            seen.add(lc)
            unique.append(lc)

    logger.info("Probing %d candidate deployments …", len(unique))

    # Probe in batches of 5
    results: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(5)

    async def _bounded(name: str) -> Dict[str, Any]:
        async with sem:
            return await _probe_one(client, name, api_key)

    tasks = [_bounded(n) for n in unique]
    results = await asyncio.gather(*tasks)

    ok     = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    # Print table
    print("\n── Deployment Discovery ──────────────────────────────────────")
    print(f"{'NAME':<35} {'STATUS':<18} {'LATENCY':>8}")
    print("─" * 65)
    for r in sorted(results, key=lambda x: (x["status"] != "ok", x["name"])):
        lat = f"{r['latency_ms']}ms" if r["latency_ms"] else "-"
        print(f"{r['name']:<35} {r['status']:<18} {lat:>8}")
    print("─" * 65)
    print(f"OK: {len(ok)}  FAILED: {len(failed)}\n")

    payload = {
        "timestamp": time.time(),
        "ok": ok,
        "failed": [{"name": r["name"], "status": r["status"]} for r in failed],
    }

    # Write results file (no secrets)
    os.makedirs("results", exist_ok=True)
    with open("results/deployments_resolved.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _CACHE = payload
    _CACHE_TS = time.time()
    return payload


def resolve_roles(
    ok_deployments: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """
    Map each role to the best available deployment name.
    Reads env vars first; falls back to preference order.
    """
    ok_names = [d["name"] for d in ok_deployments]

    def _pick_by_prefs(prefs: List[str]) -> Optional[str]:
        for pref in prefs:
            for name in ok_names:
                if pref.lower() in name.lower():
                    return name
        return ok_names[0] if ok_names else None

    roles: Dict[str, Optional[str]] = {}
    env_map = {
        "STRUCTURING":  os.getenv("STRUCTURING_DEPLOYMENT", ""),
        "EXTRACTION":   os.getenv("EXTRACTION_DEPLOYMENT", ""),
        "TIER_SIMPLE":  os.getenv("TIER_SIMPLE", ""),
        "TIER_DEFAULT": os.getenv("TIER_DEFAULT", ""),
        "TIER_ESCALATE":os.getenv("TIER_ESCALATE", ""),
    }

    for role, env_val in env_map.items():
        if env_val and env_val in ok_names:
            roles[role] = env_val
            logger.info("Role %s → %s (from env)", role, env_val)
        elif env_val and env_val not in ok_names:
            # Configured but not working – warn, do not silently swap
            logger.warning(
                "Role %s configured as '%s' but that deployment is NOT working. "
                "Falling back to auto-selection.",
                role, env_val,
            )
            chosen = _pick_by_prefs(ROLE_PREFS[role])
            roles[role] = chosen
            logger.info("Role %s → %s (auto fallback)", role, chosen)
        else:
            chosen = _pick_by_prefs(ROLE_PREFS[role])
            roles[role] = chosen
            logger.info("Role %s → %s (auto)", role, chosen)

    return roles


def get_cached() -> Optional[Dict[str, Any]]:
    return _CACHE


def invalidate_cache() -> None:
    global _CACHE, _CACHE_TS
    _CACHE = None
    _CACHE_TS = 0.0
