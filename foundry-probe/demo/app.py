"""
demo/app.py – FastAPI application.
Python 3.9 compatible.
SECURITY: keys never sent to browser; all errors redacted; no patient data logged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import openai
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import (
    FOUNDRY_API_KEY, ROUTER_MODE, budget_manager, client,
    MUNSIT_API_KEY, MISTRAL_API_KEY,
)
from demo.deployments import discover, resolve_roles, get_cached, invalidate_cache
from demo.errors import redact as _redact, map_provider_error as _map_err
from demo.model_profiles import load_from_resolved
from demo.pipeline import (
    term_correction, run_structuring, run_extraction, run_reasoning,
)
from demo.router import evaluate_triggers, determine_initial_tier, determine_escalation, log_audit
from demo.stt import STTProvider, convert_audio_to_pcm16_16k
from demo.validation import validate_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedVoice Demo")

# Role map populated at startup
_ROLES: Dict[str, Optional[str]] = {
    "STRUCTURING":  None,
    "EXTRACTION":   None,
    "TIER_SIMPLE":  None,
    "TIER_DEFAULT": None,
    "TIER_ESCALATE":None,
}
_EXTRA_NAMES: list = []


def _provider_error(exc: Exception, step: str, deployment: Optional[str]) -> JSONResponse:
    """Map provider errors to safe JSON responses via demo.errors."""
    mapped = _map_err(exc, step, deployment)
    return JSONResponse(status_code=mapped["status_code"], content=mapped["body"])


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    global _ROLES, _EXTRA_NAMES
    logger.info("Running deployment discovery …")

    for var in ("DEPLOYMENTS", "STRUCTURING_DEPLOYMENT", "EXTRACTION_DEPLOYMENT",
                "TIER_SIMPLE", "TIER_DEFAULT", "TIER_ESCALATE"):
        val = os.getenv(var, "")
        if "," in val:
            _EXTRA_NAMES.extend([v.strip() for v in val.split(",") if v.strip()])
        elif val:
            _EXTRA_NAMES.append(val)

    try:
        result = await discover(client, FOUNDRY_API_KEY, extra_names=_EXTRA_NAMES)
        _ROLES = resolve_roles(result["ok"])
        load_from_resolved()
        logger.info("Startup discovery complete. Roles: %s", _ROLES)
    except Exception as exc:
        logger.error("Startup discovery failed: %s", type(exc).__name__)

    logger.info("MedVoice Demo ready on http://0.0.0.0:8123")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def get_index() -> HTMLResponse:
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/health")
async def health() -> JSONResponse:
    cached = get_cached()
    ok_names = [d["name"] for d in cached["ok"]] if cached else []
    failed = cached["failed"] if cached else []
    return JSONResponse({
        "status": "ok" if _ROLES.get("STRUCTURING") else "degraded",
        "roles": _ROLES,
        "ok_deployments": ok_names,
        "failed_deployments": failed,
        "budget_spent_usd": round(budget_manager.current_spend, 6),
        "python_version": sys.version,
        "openai_version": openai.__version__,
    })


@app.post("/api/refresh-deployments")
async def refresh_deployments() -> JSONResponse:
    invalidate_cache()
    result = await discover(client, FOUNDRY_API_KEY, extra_names=_EXTRA_NAMES)
    _ROLES.update(resolve_roles(result["ok"]))
    load_from_resolved()
    return JSONResponse({"roles": _ROLES, "ok_count": len(result["ok"])})


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(...),
) -> JSONResponse:
    import tempfile
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    pcm_path = tmp_path + "_pcm16.wav"

    try:
        await convert_audio_to_pcm16_16k(tmp_path, pcm_path)
    except Exception as exc:
        logger.error("FFmpeg conversion error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    stt = STTProvider()
    result = await stt.transcribe(pcm_path, language)
    try: os.unlink(pcm_path)
    except OSError: pass

    return JSONResponse(result)


class ProcessRequest(BaseModel):
    transcript: str
    department: str
    history_text: str = ""
    mode: str = ROUTER_MODE
    in_depth_review: bool = False


@app.post("/api/process")
async def process_pipeline(req: ProcessRequest) -> JSONResponse:
    struct_dep  = _ROLES.get("STRUCTURING")
    extract_dep = _ROLES.get("EXTRACTION")

    if not struct_dep:
        return JSONResponse(status_code=503, content={
            "error": "NoDeployment",
            "message": "No working deployment found for STRUCTURING. Check /api/health.",
            "step": "startup",
        })

    # 1. Term correction (deterministic, never fails)
    corrected, corrections, tc_lat = term_correction(req.transcript)

    # 2. Structuring
    try:
        structured_text, signals, struct_lat = await run_structuring(corrected, struct_dep)
    except Exception as exc:
        return _provider_error(exc, "structuring", struct_dep)

    # 3. Router – initial tier
    triggers = evaluate_triggers(signals, req.history_text)
    initial_tier_dep, tier_reason = determine_initial_tier(req.mode, triggers, req.in_depth_review, _ROLES)

    # Resolve role name → actual deployment name
    def _role_dep(role_name: Optional[str]) -> Optional[str]:
        if not role_name:
            return None
        for role_key, dep_name in _ROLES.items():
            if dep_name == role_name:
                return dep_name
        # initial_tier_dep IS already a deployment name from config
        return role_name

    reasoning_dep = initial_tier_dep if initial_tier_dep else _ROLES.get("TIER_DEFAULT")
    extract_dep   = extract_dep or _ROLES.get("EXTRACTION") or struct_dep

    # 4. Parallel extraction + reasoning
    try:
        (extraction_json, ext_lat), (reasoning_json, reas_lat) = await asyncio.gather(
            run_extraction(structured_text, req.department, extract_dep),
            run_reasoning(structured_text, req.history_text, reasoning_dep),
        )
    except Exception as exc:
        return _provider_error(exc, "extraction/reasoning", reasoning_dep)

    # 5. Validation
    cautions = validate_payload(extraction_json, reasoning_json, req.history_text, None)

    # 6. Escalation
    escalation_dep, esc_reason = determine_escalation(
        req.mode, initial_tier_dep, cautions, req.in_depth_review, reasoning_json, _ROLES
    )
    if escalation_dep and escalation_dep != initial_tier_dep:
        esc_real = _ROLES.get("TIER_ESCALATE")
        if esc_real:
            try:
                reasoning_json, esc_lat = await run_reasoning(
                    structured_text, req.history_text, esc_real
                )
                reas_lat += esc_lat
                cautions = validate_payload(extraction_json, reasoning_json, req.history_text, None)
            except Exception as exc:
                cautions.append(f"Escalation failed ({type(exc).__name__}); using original reasoning.")
                esc_reason = "escalation_failed"
        else:
            cautions.append("TIER_ESCALATE not available; escalation skipped.")
            esc_reason = "no_escalate_deployment"

    audit: Dict[str, Any] = {
        "mode": req.mode,
        "signals": signals,
        "triggers": triggers,
        "initial_tier": initial_tier_dep,
        "initial_reason": tier_reason,
        "escalation_tier": escalation_dep,
        "escalation_reason": esc_reason,
        "deployments": {
            "structuring":  struct_dep,
            "extraction":   extract_dep,
            "reasoning":    reasoning_dep,
        },
        "latencies": {
            "term_correction_s": round(tc_lat, 3),
            "structuring_s":     round(struct_lat, 3),
            "extraction_s":      round(ext_lat, 3),
            "reasoning_s":       round(reas_lat, 3),
        },
        "cost_usd": round(budget_manager.current_spend, 6),
    }
    log_audit(audit)

    return JSONResponse({
        "corrected_transcript": corrected,
        "corrections": corrections,
        "structured_text": structured_text,
        "extraction": extraction_json,
        "reasoning": reasoning_json,
        "cautions": cautions,
        "audit": audit,
    })


@app.exception_handler(Exception)
async def global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "message": "Internal server error."},
    )
