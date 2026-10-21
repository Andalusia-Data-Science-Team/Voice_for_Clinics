#!/usr/bin/env python3
"""Smoke-test the six Azure AI Foundry deployments (A3.3).

CREATE ONLY — the owner runs this. It makes live calls to Azure AI Foundry.

For each of the six deployments it runs a small batch of synthetic clinical
sentences in English and Arabic (invented here, no real patient data). For the
extraction/reasoning/router roles it validates strict JSON against the project's
schema. It counts content-filter blocks per deployment and prints ONLY counts and
pass/fail — never prompts, responses, or secrets.

Configuration is read from the environment at runtime (AI_FOUNDRY_ENDPOINT,
AI_FOUNDRY_KEY / managed identity, deployment names). No secrets are printed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Synthetic clinical sentences (invented; no real patient data).
SYNTHETIC_EN: List[str] = [
    "Synthetic visit: mild headache for two days, no red flags, vitals stable.",
    "Synthetic visit: sore throat and mild fever for three days, no difficulty breathing.",
    "Synthetic visit: left knee pain after a minor fall, no swelling, walks normally.",
    "Synthetic visit: intermittent cough for one week, non-smoker, no chest pain.",
    "Synthetic visit: routine follow-up for controlled hypertension, feels well.",
    "Synthetic visit: mild lower back pain after lifting, no numbness or weakness.",
    "Synthetic visit: seasonal allergic rhinitis, sneezing and itchy eyes.",
    "Synthetic visit: minor ankle sprain, able to bear weight, no deformity.",
    "Synthetic visit: occasional heartburn after meals, no weight loss.",
    "Synthetic visit: mild fatigue for several days, sleeping and eating normally.",
]

SYNTHETIC_AR: List[str] = [
    "زيارة اصطناعية: صداع خفيف منذ يومين، لا علامات خطيرة، العلامات الحيوية مستقرة.",
    "زيارة اصطناعية: التهاب حلق وحمى خفيفة منذ ثلاثة أيام، لا ضيق في التنفس.",
    "زيارة اصطناعية: ألم في الركبة اليسرى بعد سقوط بسيط، لا تورم، يمشي بشكل طبيعي.",
    "زيارة اصطناعية: سعال متقطع منذ أسبوع، غير مدخن، لا ألم في الصدر.",
    "زيارة اصطناعية: متابعة روتينية لارتفاع ضغط الدم المضبوط، يشعر بحالة جيدة.",
    "زيارة اصطناعية: ألم خفيف أسفل الظهر بعد حمل شيء، لا تنميل أو ضعف.",
    "زيارة اصطناعية: حساسية موسمية، عطاس وحكة في العينين.",
    "زيارة اصطناعية: التواء بسيط في الكاحل، يستطيع تحمل الوزن، لا تشوه.",
    "زيارة اصطناعية: حرقة معدة متقطعة بعد الأكل، لا فقدان وزن.",
    "زيارة اصطناعية: إرهاق خفيف منذ عدة أيام، النوم والأكل طبيعيان.",
]

# Roles that must return strict JSON.
JSON_ROLES = {"extraction", "reasoning", "router"}

ROUTER_SCHEMA_KEYS = {"complexity", "reason_codes"}


def _deployments() -> List[Tuple[str, str]]:
    """Return (role, deployment) pairs from the environment."""
    return [
        ("refinement", os.getenv("FOUNDRY_MODEL_REFINEMENT", "DeepSeek-V3.2")),
        ("translation", os.getenv("FOUNDRY_MODEL_TRANSLATION", "DeepSeek-V3.2")),
        ("extraction", os.getenv("FOUNDRY_MODEL_EXTRACTION", "gpt-5.6-luna")),
        ("extraction_fallback", os.getenv("FOUNDRY_MODEL_EXTRACTION_FALLBACK", "gpt-4.1-mini")),
        ("router", os.getenv("FOUNDRY_MODEL_ROUTER", "model-router")),
        ("reasoning_simple", os.getenv("FOUNDRY_MODEL_REASONING_SIMPLE", "gpt-5.6-luna")),
        ("reasoning_complex", os.getenv("FOUNDRY_MODEL_REASONING_COMPLEX", "gpt-5.6-sol")),
        ("reasoning_simple_fallback", os.getenv("FOUNDRY_MODEL_REASONING_SIMPLE_FALLBACK", "gpt-4.1-mini")),
        ("reasoning_complex_fallback", os.getenv("FOUNDRY_MODEL_REASONING_COMPLEX_FALLBACK", "gpt-5.6-terra")),
    ]


def _is_content_filter(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    if code == "content_filter":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict) and err.get("code") == "content_filter":
            return True
    return False


async def _run_deployment(client, role: str, deployment: str) -> Dict[str, Any]:
    """Run the synthetic batch against one deployment; return counts only."""
    from llm.provider import build_foundry_kwargs

    sentences = SYNTHETIC_EN + SYNTHETIC_AR
    ok = 0
    json_ok = 0
    json_bad = 0
    content_filter_blocks = 0
    errors = 0

    for sentence in sentences:
        wants_json = role in JSON_ROLES
        kwargs = build_foundry_kwargs(
            deployment,
            messages=[{"role": "user", "content": sentence}],
            max_tokens=512,
            stream=False,
            response_format={"type": "json_object"} if wants_json else None,
        )
        try:
            resp = await client.chat.completions.create(**kwargs)
            ok += 1
            if wants_json:
                content = ""
                if resp.choices and resp.choices[0].message.content:
                    content = resp.choices[0].message.content
                try:
                    parsed = json.loads(content)
                    if role == "router":
                        valid = isinstance(parsed, dict) and ROUTER_SCHEMA_KEYS.issubset(parsed)
                    else:
                        valid = isinstance(parsed, dict)
                    json_ok += 1 if valid else 0
                    json_bad += 0 if valid else 1
                except Exception:
                    json_bad += 1
        except Exception as exc:
            if _is_content_filter(exc):
                content_filter_blocks += 1
            else:
                errors += 1

    return {
        "role": role,
        "deployment": deployment,
        "sentences": len(sentences),
        "ok": ok,
        "json_ok": json_ok,
        "json_bad": json_bad,
        "content_filter_blocks": content_filter_blocks,
        "errors": errors,
        "pass": errors == 0 and content_filter_blocks == 0 and json_bad == 0,
    }


async def main() -> int:
    """Run the smoke test. Prints counts and pass/fail only."""
    from openai import AsyncOpenAI

    endpoint = os.getenv("AI_FOUNDRY_ENDPOINT") or os.getenv("FOUNDRY_BASE_URL")
    if not endpoint:
        print("AI_FOUNDRY_ENDPOINT is not set; cannot run.", file=sys.stderr)
        return 2
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"

    api_key = os.getenv("AI_FOUNDRY_KEY") or os.getenv("FOUNDRY_API_KEY") or os.getenv("AZURE_API_KEY")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=45.0)

    results = []
    for role, deployment in _deployments():
        results.append(await _run_deployment(client, role, deployment))

    await client.close()

    print("role,deployment,sentences,ok,json_ok,json_bad,content_filter_blocks,errors,pass")
    all_pass = True
    for r in results:
        all_pass = all_pass and r["pass"]
        print(
            f"{r['role']},{r['deployment']},{r['sentences']},{r['ok']},"
            f"{r['json_ok']},{r['json_bad']},{r['content_filter_blocks']},{r['errors']},"
            f"{'PASS' if r['pass'] else 'FAIL'}"
        )
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))