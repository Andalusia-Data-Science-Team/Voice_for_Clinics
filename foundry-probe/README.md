# MedVoice foundry-probe

Local demo and evaluation suite for the MedVoice medical dictation pipeline on Azure AI Foundry.

## Quick Start

```bash
cd foundry-probe
cp .env.example .env        # fill in your real keys
source .venv/bin/activate
pip install -r requirements.txt

# 1. Discover which deployments are actually live
python check_deployments.py

# 2. Start the demo server
uvicorn demo.app:app --host 0.0.0.0 --port 8123

# 3. Open http://localhost:8123
```

## Deployment Discovery

At startup, `demo/app.py` automatically runs `demo/deployments.py:discover()` which:

1. Collects candidate names from:
   - `DEPLOYMENTS` env var (comma-separated)
   - Per-role env vars (`STRUCTURING_DEPLOYMENT`, `EXTRACTION_DEPLOYMENT`, `TIER_SIMPLE`, `TIER_DEFAULT`, `TIER_ESCALATE`)
   - A built-in list of likely names (`gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, `gpt-4o-mini`, DeepSeek variants, etc.)
   - Any names returned by `client.models.list()` that match known GPT/DeepSeek/Luna/Terra/Sol patterns

2. Probes each candidate with a tiny one-word request (max 16 tokens), up to 5 in parallel, with a 20s timeout.

3. Classifies each as: `ok`, `not_found`, `auth_error`, `rate_limited`, `timeout`, or `error:…`

4. Writes `results/deployments_resolved.json` (no secrets) with capabilities per working deployment.

5. Caches results for 10 minutes. Refresh via `POST /api/refresh-deployments`.

### Capability Probing

For each `ok` deployment, it tests:
- `max_completion_tokens` vs `max_tokens` (falls back automatically)
- `streaming` support
- `json_object` response format
- `json_schema_strict` structured outputs (falls back to `json_object` + local validation if unsupported)
- `temperature` parameter support
- `reasoning_effort` support (o-series / Sol-class models)

## Overriding Roles in .env

```env
# Explicitly assign a deployment to a role (name must be in the working set)
STRUCTURING_DEPLOYMENT=gpt-5.6-luna
EXTRACTION_DEPLOYMENT=gpt-4o-mini
TIER_SIMPLE=gpt-5.6-luna
TIER_DEFAULT=gpt-5.6-terra
TIER_ESCALATE=gpt-5.6-sol

# Router mode: difficulty (default) or v34
ROUTER_MODE=difficulty

# STT budget guard
MAX_STT_MINUTES=10
```

If an env-configured deployment is not in the working set, the system logs a warning and auto-selects a fallback. It **never** silently uses a different model when the configured one is working.

## Reading /api/health

```
GET http://localhost:8123/api/health
```

Returns:
```json
{
  "status": "ok|degraded",
  "roles": {
    "STRUCTURING": "gpt-5.6-luna",
    "EXTRACTION":  "gpt-4o-mini",
    "TIER_SIMPLE": "gpt-5.6-luna",
    "TIER_DEFAULT": "gpt-5.6-terra",
    "TIER_ESCALATE": "gpt-5.6-sol"
  },
  "ok_deployments": ["gpt-5.6-luna", "gpt-5.6-terra", ...],
  "failed_deployments": [{"name": "gpt-6-sol", "status": "not_found"}, ...],
  "budget_spent_usd": 0.000123,
  "python_version": "3.9.x ...",
  "openai_version": "2.x.x"
}
```

The UI shows a red banner and disables the **Process Pipeline** button when any required role has no deployment.

## Running Tests (no network, no cost)

```bash
PYTHONPATH=. pytest tests/ -v
```

Covers: router logic, validation rules, discovery with fake client, role resolution, parameter profile fallbacks, JSON schema downgrade, key-never-leaked security.

## Architecture

```
demo/
  app.py            – FastAPI app (startup discovery, routes, error handling)
  deployments.py    – Discovery + role resolution (cached 10 min)
  model_profiles.py – Per-deployment capability profiles + safe kwargs builder
  pipeline.py       – Term correction, structuring, extraction, reasoning
  router.py         – difficulty / v34 routing logic
  validation.py     – Deterministic clinical validation rules
  errors.py         – Provider error mapping (no secrets)
  stt.py            – Munsit + Mistral batch STT
  index.html        – Dark-theme single-page UI

probes/
  stt_stream_munsit.py   – WebSocket streaming probe for Munsit
  stt_stream_mistral.py  – Placeholder for Mistral realtime

check_deployments.py     – Standalone discovery CLI
eval_router.py           – Router evaluation over synthetic + cases.json cases
```
