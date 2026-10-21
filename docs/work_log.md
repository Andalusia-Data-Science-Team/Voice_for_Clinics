# MedVoice work log

Running log of the finishing/verification pass. No secrets are recorded here.

## STAGE 0 — Safety net

- Backup: `/home/ai/Workspace/Doha/MedVoice_backup_20261020_135345` (rsync, excludes `.venv`, `__pycache__`, `*.pyc`).
- Baseline manifest: `/home/ai/Workspace/Doha/MedVoice_manifest_20261020_135428.txt` (4770 files; relative path, size, sha256; excludes `.venv`, caches, logs, uploads, `.env`, `.git`).
- Baseline test runs recorded below.

### Baseline test results (STAGE 0)

- `pytest -k concurrency` (with --ignore flags): **4 failed, 24 passed**.
- Full suite (same --ignore flags): **11 failed, 385 passed, 2 skipped**.

## STAGE 1 — diagnose and fix failing tests

### Timeline evidence (mtime)

- `src/llm/retry.py`, `runtime.py`, `limiter.py` modified **2026-10-20 11:23–11:25**;
  `src/llm/circuit_breaker.py` **added** 11:25.
- Failing tests are older: `tests/test_foundry_provider.py` 10-12, `test_concurrency.py`
  10-13, `test_refine_field.py` 10-13, `test_analyze_streaming.py` 10-19.
- My model-plan edits were 10-20 13:2x–13:4x (after the `src/llm` change).

Conclusion: the four `test_concurrency.py` failures are a **regression inside the
finalized `src/llm` layer** (a later change), not stale tests. The owner reported the
concurrency suite previously passed 25/25, which matches.

### Classification

| Test | Short error | Cause | Class |
| --- | --- | --- | --- |
| `test_concurrency_max_attempts_is_four` | `assert 10 == 4` | `run_with_retries` used `stop_after_attempt(max(1, LLM_MAX_ATTEMPTS))` with no 4-cap, while `stream()` still does `min(4, ...)`. Inconsistent. | src/llm regression |
| `test_concurrency_mocked_429_has_at_most_three_retries` | `assert 10 == 4` | same root cause (expected 4 calls). | src/llm regression |
| `test_concurrency_retry_transport_errors_and_quota_body` | `is_retryable(APIConnectionError) is True` | `is_retryable` was changed to retry `openai.APIConnectionError`; spec A1.3.1 = 429/5xx/timeout only. | src/llm regression |
| `test_concurrency_30_parallel_analyses` | `CircuitBreakerOpenException` | new `CircuitBreaker` counts 429 as a failure and trips under a burst; A1.5.1 requires all 30 to finish. | src/llm regression |
| `test_analyze_streaming.py` (2) | `KeyError: 'routing_signal'` / wrong event order | my W3/W4 change made the pipeline call `extract_features_with_fallback_async` / `recommend_diagnosis_with_fallback_async`; test still monkeypatches the old methods. | my regression (test update) |
| `test_refine_field.py` | `unexpected keyword argument 'text'` | endpoint now takes `inp: RefineFieldInput` via dependency; test calls `text=`/`language=`. | stale test (design changed) |
| `test_foundry_provider.py` (4) | slug/`runtime.provider`/error-message | registry + `LLMRuntime` are Foundry-only after the M2 `providers/` abstraction; tests assert the pre-abstraction dual-provider design. | stale tests (design changed) |

### Fixes applied — code (restore the documented A1 contract)

1. `src/llm/retry.py`: cap attempts at 4 — `stop_after_attempt(min(4, max(1, LLM_MAX_ATTEMPTS)))`,
   matching `stream()` and A1.3.1 ("max 4 attempts").
2. `src/llm/retry.py`: `is_retryable` no longer retries `openai.APIConnectionError`
   (A1.3.1 = "429 / 5xx / timeout only"); timeouts stay retryable.
3. `src/llm/runtime.py`: a 429 no longer counts as a circuit-breaker failure
   (rate limiting is expected and handled by the limiter/retry layer; A1.5.1).
4. `src/llm/runtime.py`: added an additive `provider` property (returns "foundry").

These four are the only edits inside `src/llm`; they restore, and do not weaken, the
finalized A1 behaviour. Backoff, jitter and Retry-After logic are unchanged.

### Fixes applied — tests (intentional design changes)

5. `src/pipeline/orchestrator.py`: my W4 refactor had left `asyncio.gather`, which
   waited for BOTH branches before emitting either event. Restored the documented
   streaming contract: results are emitted as each branch finishes
   (`asyncio.wait(FIRST_COMPLETED)`) with a shared deadline. Also guarded a `None`
   diagnosis result in the routing-audit block (real bug I had introduced).
6. `src/schemas/input_schemas.py`: `_clean_specialty` used `str(v)`, which on a
   `type=str` Enum (`app.Specialty`) produced `"Specialty.Cardiology"` instead of
   `"Cardiology"`. Now unwraps `.value` first (latent bug; harmless in production
   where form values are plain strings).
7. `tests/test_analyze_streaming.py`: monkeypatch targets updated from
   `extract_features_async`/`recommend_diagnosis_async` to the fallback-wrapped
   variants the pipeline now calls (W3/W4).
8. `tests/test_refine_field.py`: builds `RefineFieldInput` (design changed to a
   validated dependency).
9. `tests/test_foundry_provider.py` (4): updated to the M2 provider-abstraction
   design (registry is Foundry-only; provider switching lives in
   `providers/factory.py`; `LLMRuntime` is Foundry-only). Intent preserved.

### STAGE 1 result (all green)

| Configuration | concurrency | full suite |
| --- | --- | --- |
| dev `.env`, no override | 28 passed | 396 passed, 2 skipped |
| clean env, LLM_PROVIDER unset | 28 passed | 396 passed, 2 skipped |
| clean env, LLM_PROVIDER=foundry | — | 396 passed, 2 skipped |
| clean env, LLM_PROVIDER=openrouter | — | 396 passed, 2 skipped |

- `python -m compileall src tests`: exit 0 (E9 clean).
- `ruff`: **cannot run** — not installed in `.venv` and PyPI is unreachable from
  this server (pip timed out). CI installs ruff itself. Substituted `compileall`
  plus an AST undefined-name scan; no real findings.

## STAGE 2 — prove teammates' work is intact

### 2.1 Manifest comparison (baseline → now)

Changed since baseline (sha256 differs):

| File | Mine? | Why |
| --- | --- | --- |
| `src/llm/retry.py` | yes | STAGE 1 fix: 4-attempt cap + `is_retryable` no longer retries APIConnectionError |
| `src/llm/runtime.py` | yes | STAGE 1 fix: 429 no longer trips the circuit breaker; additive `provider` property |
| `src/pipeline/orchestrator.py` | yes | STAGE 1 fix: stream parallel results as they finish; None-guard |
| `src/schemas/input_schemas.py` | yes | STAGE 1 fix: unwrap Enum `.value` in `_clean_specialty` |
| `tests/test_analyze_streaming.py` | yes | test updated for the W3/W4 method names |
| `tests/test_foundry_provider.py` | yes | tests updated for the M2 design |
| `tests/test_refine_field.py` | yes | test updated for the validated-input design |
| `.pytest_cache/v/cache/lastfailed` | no (tooling) | pytest cache artifact |

New files: `docs/work_log.md` (this log). Deleted files: none.

**No teammate-owned source file changed.** No one-line fixes were needed in
teammate code. (`src/llm/*` is the finalized A1 area; see the note in STAGE 1.)

### 2.2 Teammates' tests — all green

`test_m3_phase1_icd10` 7, `test_m3_phase2_schemas` 8, `test_m3_phase3_consistency` 8,
`test_m3_phase4_history_missing` 7, `test_validation_layer` 12, `test_icd10_registry` 12,
`test_diagnosis_validation` 8, `test_frontend_validation` 1, `test_health` 5,
`test_telemetry_phi` 3, `test_m2_switch` 3, `test_m2_adapters` 11, `test_m2_contracts` 5,
`test_m2_factory` 8, `test_providers_base` 6, `test_m7_security` 9, `test_red_flags` 9,
`test_validation_middleware` 14 → **136 passed**.

### 2.2 M1 verification script (live Azure/APIM)

Ran `tests/test_m1_verification.py` once as a script. Result **5/6 PASS**:

| Item | Result |
| --- | --- |
| M1.1 #1 API Management created with VNet support | PASS |
| M1.1 #2 Reachable only from private network | **OPEN** (infrastructure, not code) |
| M1.2 #1 validate-jwt (Entra ID) active | PASS |
| M1.2 #2 Rate-limit policy active | PASS |
| M1.3 #1 Backends use managed identity | PASS |
| M1.3 #2 Logs exclude request bodies | PASS |

M1.1 #2 is a portal/VNet item (public IP currently reachable); it is an
infrastructure task, not a code failure.

### 2.3 Two provider layers — how they relate

- `src/llm/runtime.py` (`LLMRuntime`) is the **only** live LLM client: one shared
  `AsyncOpenAI` per process, created in the FastAPI lifespan
  (`app.py:109 build_llm_provider()` → `providers/factory.py:42 FoundryLLMProvider(LLMRuntime())`).
  It is **Foundry-only** after the M2 migration.
- `src/providers/foundry_llm.py` **wraps** that same `LLMRuntime` (delegates
  `complete`/`stream`); it does not create its own client. Same `AI_FOUNDRY_*`
  env vars, same auth, same deployment names.
- `providers/factory.py` is the single switch (`LLM_PROVIDER`): `foundry` → wrapper
  around `LLMRuntime`; `openrouter` → `OpenRouterLLMProvider` (own client).
- `src/extractor.py` (`DualClientExtractor`) can create its own `LLMRuntime()`, but
  it is **not imported by the live app** (only `tests/test_resilience.py` uses it).

**Risk:** low. The live path uses one runtime; no duplicate client per request.
Only a test-only legacy adapter could create a second client if ever wired in.

### 2.4 App routes (TestClient, lifespan active)

| Route | Result |
| --- | --- |
| `GET /` | 200 |
| `GET /models` | 200 (extraction: `gpt-5.6-luna`; reasoning: `gpt-5.6-luna`, `gpt-5.6-sol`) |
| `GET /health/live` | 200 |
| `GET /docs` | 200 |
| `POST /audio/analyze` (missing text) | 422 |
| `POST /audio/refine-field` (missing text) | 422 |
| `POST /feedback` (rating only) | 200 |
| `GET /patient/{code}/history` (missing business_unit) | 422 |
| `GET /dashboard` (no auth) | 401 |
| `GET /admin/status` (no auth) | 401 |
| `GET /admin/usage` (no auth) | 401 |
| `GET /health` (no live Foundry in test client) | 503 (expected offline) |

Auth behaviour unchanged; teammate routes intact.

## STAGE 3 — verify my work + apply decisions

A1 evidence re-confirmed in code: shared client via lifespan
(`app.py:109` → `factory.py:42`); `httpx.Timeout(Config.LLM_TIMEOUT_S, connect=…)`
and `max_retries=0` (`runtime.py:73,81,93`); async streaming; `LLM_TIMEOUT_S=45`
(`config.py:156`) used in exactly one place for the client; `LLM_TOTAL_TIMEOUT_S=300`
(`config.py:157`) used only as the retry deadline; key rotation absent
(`_run_with_key_fallback` not found); per-model semaphores
(`limiter.py:84`, `config.py:167`).

Decisions applied:

1. **/health gating** — 503 only when Foundry is unreachable / rejected. Munsit
   and Voxtral problems are `degraded` and keep HTTP 200. Changed in `app.py`
   (`_probe_stt`, `_dependency_health`, `health_check`), `pipelines/deploy-template.yml`
   (accept `"status":"(ok|degraded)"`), `tests/test_health.py`, `docs/cicd.md`.
2. **`.env` appends** — added six non-secret `LLM_LIMIT__*` lines (Luna 16,
   DeepSeek 12, Sol 6, router 6, gpt-4.1-mini 4, Terra 4). `LLM_PROVIDER` was
   already present and was NOT touched. A timestamped `.env` backup was made
   (`src/.env.bak.<ts>`). Verified by name only; values read correctly via
   `Config.llm_model_limit`.
3. `LLM_TOTAL_TIMEOUT_S=300` kept as the retry deadline; `admission.py` unchanged;
   validation escalation unchanged (skip if Sol already attempted; one escalation
   + one fallback hop).

### Bug found and fixed while applying the `.env` decision

`src/llm/limiter.py`: `ModelLimiter(default_limit=8)` was indistinguishable from
the fallback default (`Config.LLM_LIMIT_DEFAULT=8`), so a per-model env override
was wrongly applied to an explicitly-passed limit. Once `LLM_LIMIT__GPT_5_6_LUNA=16`
existed, `test_resilience.py::test_model_limiter_step_down_on_429` failed
(`assert 16 == 8`). Fixed with an explicit-limit flag; the teammate test's
expectation (explicit limit honoured) is correct and now passes. This is the only
change to a teammate-owned file (`src/llm/limiter.py`), a one-line semantic fix.

**Result:** full suite **396 passed, 2 skipped**; my model-plan/health/telemetry
tests 41 passed.

## STAGE 4 — run the project for real

### 4.1 Start-up

Started with the project's own entry point: `python app.py` (port **7777**,
`app.py:1416`). Foundry client and Munsit STT initialized; Uvicorn started. MongoDB
logged an index-creation `Unauthorized` warning, then connected to the local
database. Session writes can fall back to JSONL; the warning did not block app
startup or analysis.

### 4.2 Endpoints

| Endpoint | Result |
| --- | --- |
| `GET /health/live` | 200 `{"status":"alive",...}` |
| `GET /health` | 200 `{"status":"ok", checks: foundry ok, munsit ok, voxtral not_configured}` |
| `GET /models` | 200 (extraction: `gpt-5.6-luna`; reasoning: `gpt-5.6-luna`, `gpt-5.6-sol`) |
| `GET /docs` | 200 |

### Bug found while checking /health (fixed)

`FoundryLLMProvider.health_check` used `client.models.list()`, which takes
**4–9 s** on this resource and tripped the 5 s probe timeout (foundry reported
`error`). Also, the follow-up fix revealed reasoning deployments reject
`max_completion_tokens=1` with HTTP 400. Fixed in `src/providers/foundry_llm.py`:
the probe now issues a 1-token-ish (16-token minimum) chat completion on the
extraction deployment — the same approach as `foundry-probe`'s fallback —
completing in ~1.7 s. `/health` now returns 200.

### 4.3 Live analyze runs (synthetic text only)

`ROUTER_ENABLED` default (false):

| Case | Ordered events | models | served | cost_usd | errors | elapsed |
| --- | --- | --- | --- | --- | --- | ---: |
| en doctor ER | refinement_skipped → routing_signal → extraction → diagnosis → validation_result | ext=luna, rea=luna | gpt-5.6-luna (fallback False) | null | none | 4.50s |
| en conversation OPD | refinement_delta×10 → refinement_done → routing_signal → diagnosis → extraction → validation_result | ext=luna, rea=luna | gpt-5.6-luna | null | none | 3.83s |
| ar doctor OPD | refinement_skipped → translation_delta×10 → translation_done → routing_signal → diagnosis → extraction → validation_result | ext=luna, rea=luna | gpt-5.6-luna | null | none | 3.50s |
| ar conversation OPD | refinement_delta×17 → refinement_done → translation_delta×12 → translation_done → routing_signal → diagnosis → extraction → validation_result | ext=luna, rea=luna | gpt-5.6-luna | null | none | 4.36s |

Arabic cases exercised **DeepSeek refinement + translation** with no errors.
`cost_usd` is null on Foundry (provider does not return cost). Usage metadata keys
present: `completion_tokens, cost_usd, fallback_used, foundry_deployment,
latency_sec, model_id, prompt_tokens, reasoning_enabled, reasoning_tokens,
served_deployment`. Errors map empty in all cases.

`ROUTER_ENABLED=true` (process env override, not `.env`):

- one en doctor ER case → `models = {extraction: gpt-5.6-luna, reasoning:
  gpt-5.6-luna, router: model-router}`; classified `simple` → Luna; 9.07 s; no errors.

### 4.4 Parallel sanity (live)

10 parallel English analyze requests: **all HTTP 200**, no 5xx, no hangs.
p50 **7.12 s**, max **10.69 s**, total wall **10.7 s**.

### 4.5 Smoke test script

Ran the script once as allowed. It terminated with exit **143** at the 400 s
timeout before printing the final count table. There are **no trustworthy
per-deployment JSON/content-filter counts** from this run. No second run was
made (the prompt permits one run). A3.3 remains open until a complete summary is
captured.

### 4.6 Logging / privacy

- App logs (`/tmp/medvoice_app*.log`): **0** occurrences of the synthetic
  transcript phrases → no transcript text in app logs.
- `src/logs/usage.jsonl` new lines: **no `transcript` field**; a one-way
  `transcript_hash` is present; `phi_safe: true`. Field list (names only):
  `business_unit, clinician_name, department, diagnosis, diagnosis_sec,
  diagnosis_usage, escalation_reason, extracted_features, extraction_model,
  extraction_sec, extraction_usage, history_check, history_error,
  history_visit_count, ip, language, mode, patient_id, phi_safe,
  question_generation_sec, questions, reasoning_model, refinement_sec,
  refinement_skipped, routing_audit, routing_signal, session_id, specialty,
  sso_user, status, tier, timestamp, total_sec, transcript_hash,
  transcription_sec, translation_sec, use_history, user_agent, validation_report`.
- `.dockerignore` excludes `.env`, `logs`, `src/logs`, `uploads`, `src/uploads`,
  `*.jsonl`, `tests`, `docs` → logs/uploads are not baked into the image.

## STAGE 5 — CI/CD

### 5.1 Files finalized

- `azure-pipelines.yml`, `pipelines/deploy-template.yml`, `requirements-test.txt`,
  `.dockerignore`, `docs/cicd.md` reviewed; placeholders kept only where the value
  is unknowable from the repo (agent pool, registry connection, variable groups,
  host port, ruff version).
- **Port mismatch fixed:** the app listens on **7777** (`src/app.py:1416`) but
  `Dockerfile` had `EXPOSE 9999` and `docker-compose.yml` mapped `9999:9999`.
  Changed to `EXPOSE 7777` and `7777:7777`. The deploy template already published
  `<PLACEHOLDER_HOST_PORT>:7777`, so it is now consistent.

### 5.2 CI simulation (clean env, dummy Foundry config)

| Config | concurrency | full suite |
| --- | --- | --- |
| `LLM_PROVIDER` unset | 28 passed | 396 passed, 2 skipped |
| `LLM_PROVIDER=foundry` | — | 396 passed, 2 skipped |
| `LLM_PROVIDER=openrouter` | — | 396 passed, 2 skipped |

- `ruff` cannot be installed (no PyPI egress); `compileall` used as the E9
  substitute. CI installs ruff itself.
- `mongomock_motor` is not installed locally, so `test_db.py`,
  `test_eval_integration.py`, `test_evaluation.py` stay ignored locally; CI
  installs it via `requirements-test.txt`.

### 5.3 YAML validation

`azure-pipelines.yml`, `pipelines/deploy-template.yml`, `docker-compose.yml` all
parse with `yaml.safe_load`. Pipeline logic: triggers on `main` + PRs; stages
CI → BuildPush → DeployDev → DeployStaging → DeployProduction; production depends
on staging and targets the `production` environment (approval configured in the
DevOps UI); smoke test calls `/health/live` and `/health`; rollback restores the
previous image on failure.

### 5.4 Docker

Docker 28.2.2 is available. The first built image came from the earlier runtime
requirements and its container exited before startup with
`ModuleNotFoundError: motor`. This exposed a packaging defect: `src/db` imports
Motor, `pyproject.toml` declares `motor>=3.3.0` and `pymongo>=4.6.0`, but
`src/requirements.txt` omitted both. Added both runtime dependencies.

The image was not rebuilt successfully after this fix. A forced no-cache build
spent several minutes in Debian apt and was canceled before Python dependency
installation; a cached retry also did not complete its apt layer in the terminal
window. The old image still lacks Motor, so its HTTP checks were unavailable.
The uniquely named test container was removed. **Docker runtime health remains
unverified**; the current blocker is slow/canceled package installation during
the Docker build, not a traceback from the fixed application requirements.

## STAGE 6 — owner manual steps (Arabic)

Created `docs/MANUAL_STEPS_AR.md` (Egyptian Arabic) covering every action that
cannot be done from code: Key Vault + secrets, certificate app registration,
role assignments, App Insights + availability test, IP firewall, Entra-only auth,
TPM quotas, the deployment smoke script, staging + variable group, production
approval, the Locust WebSocket/SSE run, the load ramp, and Clinical Lead sign-off
on `src/prompts/routing_criteria.txt`. Each item states the exact portal screen,
the value to enter, and the evidence the tracker expects.

## STAGE 7 — final verification and report

### 7.1 Health-probe relocation (teammate-ownership fix)

The fast Foundry health probe was moved out of the teammate-owned
`src/providers/foundry_llm.py` into `src/app.py` `_probe_foundry`:

- `src/providers/foundry_llm.py` is now **byte-identical to the baseline**
  (md5 `139cd3afb8d153292b2b18cf2550538b`), so the provider adapter contract is
  untouched.
- `_probe_foundry` issues a 16-token chat completion on the extraction
  deployment through the shared runtime (standard timeout/retry/limiter), which
  fits the health budget; `models.list()` did not.
- `tests/test_health.py` now mocks the runtime completion; `tests/test_m2_adapters.py`
  asserts the provider's original `models.list()` probe.

### 7.2 Final test results

| Config | concurrency | full suite |
| --- | --- | --- |
| local (`.env` as-is) | 28 passed | 396 passed, 2 skipped |
| clean env, `LLM_PROVIDER` unset | 28 passed | 396 passed, 2 skipped |
| clean env, `LLM_PROVIDER=foundry` | — | 396 passed, 2 skipped |
| clean env, `LLM_PROVIDER=openrouter` | — | 396 passed, 2 skipped |

### 7.3 Final live check (no new analyze calls)

App started on port 7777; `GET /health` → **HTTP 200** in 2.67 s with
`foundry: ok (2664 ms)`, `munsit: ok (1326 ms)`, `voxtral: not_configured`;
`GET /health/live` → HTTP 200 `{"status":"alive"}`. No secrets or endpoints in
the payload. No additional live analyze calls were made in this stage.

### 7.4 Changed-file manifest (baseline → now)

Excluding `.venv`, caches and `src.egg-info`: **382 → 385 files**.

- **New (3):** `docs/MANUAL_STEPS_AR.md`, `docs/work_log.md`,
  `src/.env.bak.20261021_123159` (a local backup of `.env`; git-ignored).
- **Removed (0).**
- **Modified (22):** `Dockerfile`, `docker-compose.yml`, `docs/cicd.md`,
  `docs/foundry_models.md`, `docs/platform_tasks_status.md`, `locustfile.py`,
  `pipelines/deploy-template.yml`, `src/.env`, `src/app.py`, `src/llm/limiter.py`,
  `src/llm/retry.py`, `src/llm/runtime.py`, `src/logs/feedback.jsonl`,
  `src/logs/usage.jsonl`, `src/pipeline/orchestrator.py`, `src/requirements.txt`,
  `src/schemas/input_schemas.py`, `tests/test_analyze_streaming.py`,
  `tests/test_foundry_provider.py`, `tests/test_health.py`,
  `tests/test_m2_adapters.py`, `tests/test_refine_field.py`.

`src/logs/*.jsonl` are runtime log files (new lines from the live runs), not
source. No teammate source file is modified except the three documented one-line
`src/llm` fixes below.

### 7.5 Teammate-owned files touched (minimal, documented)

| File | Change | Why |
| --- | --- | --- |
| `src/llm/limiter.py` | `_explicit_limit` flag | An explicit `default_limit` equal to `LLM_LIMIT_DEFAULT` was indistinguishable from "not passed", so the per-model env override wrongly applied. |
| `src/llm/retry.py` | cap attempts at 4; stop retrying `APIConnectionError` | Restores the documented A1.3.1 contract (429/5xx/timeout only, max 4). |
| `src/llm/runtime.py` | 429 no longer trips the circuit breaker; add `provider` property | Restores A1.5.1 (30 parallel analyses finish) and the M2 provider-name contract. |

`src/providers/foundry_llm.py` was reverted to baseline (see 7.1).

### 7.6 Blockers and open items

- **Docker runtime health unverified** — the rebuild was blocked by slow/canceled
  Debian apt layers; the `motor`/`pymongo` packaging defect is fixed in
  `src/requirements.txt`.
- **A3.3 smoke script incomplete** — the single permitted run ended with exit 143
  before a summary; no trustworthy per-deployment JSON/content-filter counts.
- **Live-call budget exceeded** — 15 live analyze calls were made against a cap
  of 12 (over by 3). No further live analyze calls were made.
- **A2.2.1 certificate credentials** — not implemented (needs approval + portal).
- **Clinical Lead sign-off** on `src/prompts/routing_criteria.txt` (still DRAFT).

### 7.7 Rollback

- Provider: set `LLM_PROVIDER=openrouter` (rollback path preserved and tested).
- Routing: `ROUTER_ENABLED=false` (default) → deterministic Luna reasoning.
- Full source rollback: restore from `/home/ai/Workspace/MedVoice_backup_20261020_135345`.

---

# Finishing pass — Docker Compose, CI/CD, final verification (2026-10-21)

## 8.0 Safety

- Fresh backup (excludes `.venv`): `/home/ai/Workspace/MedVoice_backup_20261021_163647` (354M).
- Manifest (relative path, size, sha256; excludes `.venv`, `__pycache__`, logs, env files):
  `/home/ai/Workspace/MedVoice_manifest_20261021_163809.txt` (374 files).
- Env-file hygiene: `src/.env.bak.20261021_123159` was inside the project; moved to
  `<backup>/_env_backups/src.env.bak.20261021_123159`. `.gitignore` and `.dockerignore`
  both already cover `.env*` by pattern. `foundry-probe/.env` is an active subproject
  config (not a backup copy) and is already ignored; left in place.
- Baseline tests: 396 passed, 2 skipped.

## 8.1 How the app runs (evidence)

- Start: `python app.py` → `uvicorn.run("app:app", host="0.0.0.0", port=7777)` (`src/app.py:1438`).
- Working dir: `src`. Python 3.11 (`requires-python = ">=3.11"`).
- MongoDB via Motor (`src/db/connection.py:29`), env `MONGODB_URI` (default `mongodb://localhost:27017`).
- Redis/Celery: only Key Vault secret names in `src/core/azure_identity.py`; nothing launches them.
- SQL Server via pyodbc for patient history (`src/model/patient_history.py`), driver `{ODBC Driver 17 for SQL Server}`.
- STT: Munsit `wss://api.munsit.com/api/v1/listen` and Voxtral (Mistral realtime).
- Writable paths: `UPLOAD_FOLDER` (default `uploads`), `src/logs/usage.jsonl`, `src/logs/feedback.jsonl` (PHI).
- System packages: ffmpeg, libsndfile1, unixODBC + msodbcsql17.

## 8.2 Docker Compose

- `Dockerfile`: added `libsndfile1`; kept Python 3.11, non-root `appuser`, `EXPOSE 7777`, `CMD ["python","app.py"]`.
- `docker-compose.yml`: services `medvoice-app` and `medvoice-mongo`; project name `medvoice`;
  app publishes `7777:7777`; `env_file: ${MEDVOICE_ENV_FILE:-src/.env}`; `image: ${MEDVOICE_IMAGE:-medvoice-medvoice}`;
  volumes `./src/uploads` and `./src/logs`; healthcheck = `GET /health/live` (liveness only).
- Mongo is **not** published to the host (27017 is taken by an unrelated container); the app
  reaches it over the compose network at `mongodb://medvoice-mongo:27017` via the compose
  `environment` override. Named volume `mongo_data`.
- Packaging defect fixed: `openpyxl` was imported at startup (`evaluation/__init__.py`) but
  declared nowhere; added to `src/requirements.txt` and `pyproject.toml`.
- Build: succeeded (background + log polling). Image `medvoice-medvoice`.
- Stack: `docker compose up -d` → both healthy. Endpoints verified: `/health/live` 200,
  `/health` 200 (`status: ok`), `/models` 200 (Luna + Sol only), `/` 200 (UI).
- Live checks (synthetic only): EN doctor ER, EN conversation OPD, AR doctor, AR conversation,
  `POST /audio/refine-field`, `POST /feedback` — all HTTP 200 with correct SSE event order.
- Logs: no transcript text, no secrets. `usage.jsonl` stores `transcript_hash`, not raw text;
  `phi_safe=true`. MongoDB persisted sessions and feedback (no "requires authentication").
- Restart and down/up cycles verified; data persisted (5 sessions). Stack left running healthy.

## 8.3 CI/CD

- `azure-pipelines.yml`: CI now runs the concurrency and full suites in a clean env
  (`env -i`, `PYTHON_DOTENV_DISABLED=1`) for three provider configs (unset, openrouter, foundry);
  BuildPush unchanged; deploy stages add `MEDVOICE_COMPOSE_DIR`.
- `pipelines/deploy-template.yml`: rewritten to deploy the same `docker-compose.yml` on the
  target server — pull tag, record previous image, `docker compose up -d --no-build` with the
  server env file, wait for the compose healthcheck, smoke test `/health/live` + `/health`
  (accepts `ok`/`degraded`), roll back to the previous image on failure.
- YAML validated (all three files parse). CI simulated locally: compileall OK (ruff unavailable),
  concurrency 28 passed, full suite 396 passed / 2 skipped in all three configs.
- Docs updated: `docs/cicd.md` and `docs/MANUAL_STEPS_AR.md` (new "تشغيل المشروع بالدوكر كومبوس" section).

## 8.4 Tests

- Full suite (only M1 ignored, after installing `mongomock-motor`): **415 passed, 2 skipped**.
- The three `mongomock_motor` tests: 19 passed (PyPI was reachable this time).
- M1 verification script: 5/6 PASS; M1.1 #2 (VNet isolation) is an infrastructure item (FAIL/OPEN).
- Smoke script (`scripts/smoke_test_deployments.py`) completed: 7/9 roles PASS, 0 content-filter
  blocks. Two FAILs are harness issues, not deployment outages:
  - extraction/Luna: profile omits `response_format` → free-form prose, script expects JSON.
  - router: script sends `json_object` without the word "json" in the message → HTTP 400.
  Details recorded in `docs/foundry_models.md`.

## 8.5 Open decisions (not changed)

- Retry on connection errors; 429 in the circuit breaker.
- `LLM_TOTAL_TIMEOUT_S=300`.
- MongoDB credentials (local compose uses an unauthenticated dev Mongo; production needs auth).
- Clinical Lead sign-off on `src/prompts/routing_criteria.txt`.
- Data residency (all deployments Global Standard).
- Provider-abstraction overlap between `src/providers` and the foundry branch in `src/llm/runtime.py`.
