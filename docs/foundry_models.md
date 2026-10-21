# Azure AI Foundry models — MedVoice

Ground truth (owner): all deployments are **Global Standard**. Deployment names
are used exactly as written.

## Step → deployment → fallback

| Step | Primary deployment | Fallback deployment | Notes |
| --- | --- | --- | --- |
| Refinement | `DeepSeek-V3.2` | — | No fallback defined; failure keeps today's behaviour (error event with step name, stream ends). |
| Translation | `DeepSeek-V3.2` | — | Same as refinement. |
| Extraction | `gpt-5.6-luna` | `gpt-4.1-mini` | One fallback hop on eligible failures only. |
| Reasoning (simple tier) | `gpt-5.6-luna` | `gpt-4.1-mini` | Chosen when the router returns `simple`, or when `ROUTER_ENABLED=false`. |
| Reasoning (complex tier) | `gpt-5.6-sol` | `gpt-5.6-terra` | Chosen when the router returns `complex`, or on router failure (`ROUTER_FAILURE_TIER=complex`). |
| Router (classifier only) | `model-router` | — | Returns strict JSON `{"complexity":"simple"|"complex","reason_codes":[...]}`. Never a primary. |

`gpt-5.6-terra` is **only** Sol's fallback. `gpt-4o-mini` is not used anywhere on
Foundry. Fallback-only and router-only deployments are never selectable primaries
in `GET /models`.

## Capability profiles

Profiles live in `src/llm/provider.py` (`_FOUNDRY_PROFILES`). Unknown deployments
get a permissive profile.

| Deployment | max_completion_tokens | max_tokens | streaming | json_object | json_schema_strict | temperature | reasoning_effort | Verified? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gpt-5.6-luna` | yes | no | yes | no | yes | no | yes | probe (stale) |
| `gpt-5.6-terra` | yes | no | yes | no | yes | no | yes | probe (stale) |
| `gpt-5.6-sol` | yes | no | yes | no | yes | no | yes | probe (stale) |
| `DeepSeek-V3.2` | no | yes | yes | yes | no | yes | no | **UNVERIFIED until a live run** |
| `gpt-4.1-mini` | (permissive) | (permissive) | yes | yes | no | yes | no | **UNVERIFIED until a live run** |
| `model-router` | (permissive) | (permissive) | yes | yes | no | yes | no | **UNVERIFIED until a live run** |

> **Note:** `foundry-probe/results/deployments_resolved.json` is **stale** — it was
> created before the `DeepSeek-V3.2`, `model-router` and `gpt-4.1-mini`
> deployments existed, so it lists them as `not_found`. Do **not** treat that as
> truth. The hard-coded DeepSeek capability profile is **unverified until a live
> run**; `scripts/smoke_test_deployments.py` is what verifies it.

## Live verification (2026-10-21)

Observed through the application with synthetic text only:

| Deployment/behavior | Result | Scope |
| --- | --- | --- |
| `gpt-5.6-luna` | Verified | Extraction and simple reasoning succeeded in four live analyze cases. |
| `DeepSeek-V3.2` | Partially verified | Refinement/translation streaming succeeded in English conversation and both Arabic cases. The full capability matrix (token parameter, temperature, strict schema) remains unverified. |
| `model-router` | Partially verified | One router-enabled synthetic request returned parseable `simple` JSON and selected Luna. Other decision branches and failure behavior remain mock-tested only. |
| `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-4.1-mini` | Not live-verified | No live fallback or complex-tier request was made. |
| Deployment smoke script | Incomplete | One run was started but terminated with exit 143 before printing a summary. No per-deployment counts are claimed; do not infer pass/fail from that run. |

The live app endpoint used the same Foundry `/openai/v1` path. The smoke script
must complete successfully before any deployment's JSON/content-filter profile
is considered verified.

## Smoke script run (2026-10-21, A3.3)

`scripts/smoke_test_deployments.py` completed a full run (20 synthetic sentences
per deployment, English + Arabic). Counts only:

| role | deployment | sentences | ok | json_ok | json_bad | content_filter | errors | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| refinement | `DeepSeek-V3.2` | 20 | 20 | 0 | 0 | 0 | 0 | PASS |
| translation | `DeepSeek-V3.2` | 20 | 20 | 0 | 0 | 0 | 0 | PASS |
| extraction | `gpt-5.6-luna` | 20 | 20 | 0 | 20 | 0 | 0 | FAIL |
| extraction_fallback | `gpt-4.1-mini` | 20 | 20 | 0 | 0 | 0 | 0 | PASS |
| router | `model-router` | 20 | 0 | 0 | 0 | 0 | 20 | FAIL |
| reasoning_simple | `gpt-5.6-luna` | 20 | 20 | 0 | 0 | 0 | 0 | PASS |
| reasoning_complex | `gpt-5.6-sol` | 20 | 20 | 0 | 0 | 0 | 0 | PASS |
| reasoning_simple_fallback | `gpt-4.1-mini` | 20 | 20 | 0 | 0 | 0 | 0 | PASS |
| reasoning_complex_fallback | `gpt-5.6-terra` | 20 | 20 | 0 | 0 | 0 | 0 | PASS |

**Interpretation (both FAILs are smoke-script harness issues, not deployment
outages):**

- **extraction / `gpt-5.6-luna` (json_bad=20):** Luna's capability profile sets
  `json_object: False` and `json_schema_strict: True`, so `build_foundry_kwargs`
  omits `response_format` and the model returns free-form prose. The script then
  fails to `json.loads` it. All 20 calls succeeded (`ok=20`, `errors=0`); the
  deployment is healthy. The script's expectation of strict JSON for the
  extraction role does not match Luna's profile.
- **router / `model-router` (errors=20):** the script sends
  `response_format={"type":"json_object"}` with a message that does not contain
  the word "json", so the API rejects it with HTTP 400
  (`'messages' must contain the word 'json' ...`). A direct probe with a
  JSON-shaped prompt returned HTTP 200 and resolved to `gpt-6-luna-2026-09-22`,
  so the router deployment itself works.

**Verified live by this run:** `DeepSeek-V3.2` (refinement + translation),
`gpt-4.1-mini` (extraction fallback + reasoning-simple fallback),
`gpt-5.6-sol` (reasoning-complex), `gpt-5.6-terra` (reasoning-complex fallback),
`gpt-5.6-luna` (reasoning-simple). **Zero content-filter blocks** across all
deployments. **Still unverified:** strict-JSON extraction on Luna and the
router's JSON contract through the script harness (both need a harness fix, not
a deployment change).

## Quota table (fill from the Azure portal)

TPM/RPM are **not invented here**. Fill them from the portal, then set the
matching env var `FOUNDRY_QUOTA_TPM__<DEPLOYMENT>` (non-alphanumeric → `_`).

| Deployment | TPM | RPM | Env var name |
| --- | --- | --- | --- |
| `gpt-5.6-luna` | <TO FILL> | <TO FILL> | `FOUNDRY_QUOTA_TPM__GPT_5_6_LUNA` |
| `gpt-5.6-terra` | <TO FILL> | <TO FILL> | `FOUNDRY_QUOTA_TPM__GPT_5_6_TERRA` |
| `gpt-5.6-sol` | <TO FILL> | <TO FILL> | `FOUNDRY_QUOTA_TPM__GPT_5_6_SOL` |
| `gpt-4.1-mini` | <TO FILL> | <TO FILL> | `FOUNDRY_QUOTA_TPM__GPT_4_1_MINI` |
| `DeepSeek-V3.2` | <TO FILL> | <TO FILL> | `FOUNDRY_QUOTA_TPM__DEEPSEEK_V3_2` |
| `model-router` | <TO FILL> | <TO FILL> | `FOUNDRY_QUOTA_TPM__MODEL_ROUTER` |

## Little's law and recommended `LLM_LIMIT__<MODEL>` values

Baseline: **30 encounters/minute** (0.5/s), **8 s** latency per call.
`in_flight ≈ calls/s × latency`.

| Deployment | Calls per encounter | calls/s | in_flight ≈ | Recommended `LLM_LIMIT__<MODEL>` |
| --- | --- | --- | --- | --- |
| `DeepSeek-V3.2` (refine + translate, Arabic) | 2 | 1.0 | 8 | `LLM_LIMIT__DEEPSEEK_V3_2=8` |
| `gpt-5.6-luna` (extraction + reasoning-simple) | 2 | 1.0 | 8 | `LLM_LIMIT__GPT_5_6_LUNA=8` |
| `gpt-5.6-sol` (reasoning-complex) | 1 | 0.5 | 4 | `LLM_LIMIT__GPT_5_6_SOL=4` |
| `model-router` | 1 | 0.5 | 4 | `LLM_LIMIT__MODEL_ROUTER=4` |
| `gpt-4.1-mini` (fallback-only) | rare | — | 2 | `LLM_LIMIT__GPT_4_1_MINI=2` |
| `gpt-5.6-terra` (fallback-only) | rare | — | 2 | `LLM_LIMIT__GPT_5_6_TERRA=2` |

`LLM_LIMIT_DEFAULT` stays `8`. Limits are process-local; with multiple workers
each worker has its own semaphores.

## Manual checklist (not code)

- [ ] Azure portal: Foundry IP firewall (A2.5.1).
- [ ] Azure portal: disable local (API-key) auth; Entra ID tokens only (A2.5.2).
- [ ] Azure DevOps: staging environment + variable group (A4.2).
- [ ] Azure DevOps: production approval check (A4.3.2).
- [ ] Run `scripts/smoke_test_deployments.py` (A3.3).
- [ ] Fill the quota table above (A3.2).
- [ ] Clinical Lead sign-off on `src/prompts/routing_criteria.txt`.