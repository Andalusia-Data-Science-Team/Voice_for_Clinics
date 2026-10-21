# MedVoice — Final Report (finishing pass, 2026-10-21)

## 1. Verdict (Egyptian Arabic, ≤8 lines)

المشروع شغّال بالدوكر كومبوس فعليًا: `docker compose up -d` طلّع خدمتين (`medvoice-app` و`medvoice-mongo`) والاتنين healthy، وكل الـ endpoints (`/health/live`, `/health`, `/models`, `/`) رجّعت 200. التستات كلها خضرا: 415 نجحوا و2 متخطّيين (و396/2 في البيئة النضيفة للتلات إعدادات). اللي اتأكد live: التحليل بالعربي والإنجليزي، doctor وconversation، refine-field، feedback، والـ MongoDB بيحفظ. اللي mock-only: الـ fallbacks وبعض فروع الـ router. اللي لسه مش متأكد: strict-JSON extraction على Luna والـ router من خلال سكربت الـ smoke (مشاكل في السكربت نفسه مش في التوزيعات). فاضل على الإنسان: تشغيل الـ pipeline على Azure DevOps، وملء جدول الحصص، وتوقيع الـ Clinical Lead، وشغل البوابة حسب `docs/MANUAL_STEPS_AR.md`.

## 2. Test results

| Command / configuration | Passed | Failed | Skipped | Not run | Reason |
| --- | --- | --- | --- | --- | --- |
| Baseline: `pytest -q` (4 ignores) | 396 | 0 | 2 | 0 | — |
| Full suite, only M1 ignored (after installing `mongomock-motor`) | 415 | 0 | 2 | 0 | — |
| Clean env (`env -i`, `PYTHON_DOTENV_DISABLED=1`), `LLM_PROVIDER` unset | 396 | 0 | 2 | 0 | — |
| Clean env, `LLM_PROVIDER=openrouter` | 396 | 0 | 2 | 0 | — |
| Clean env, `LLM_PROVIDER=foundry` | 396 | 0 | 2 | 0 | — |
| Concurrency (`-k concurrency`), clean env, provider unset | 28 | 0 | 0 | 0 | — |
| Concurrency, clean env, `foundry` | 28 | 0 | 0 | 0 | — |
| `tests/test_db.py` + `test_eval_integration.py` + `test_evaluation.py` | 19 | 0 | 0 | 0 | Needed `mongomock-motor`; PyPI was reachable this run |
| `tests/test_m1_verification.py` (script) | 5/6 items | 1 item | 0 | 0 | M1.1 #2 VNet isolation = infrastructure item (FAIL/OPEN) |
| Lint selection `E9,F63,F7,F82` | — | — | — | — | `ruff` not installed locally; `compileall` fallback exit 0 |
| `scripts/smoke_test_deployments.py` | 7/9 roles | 2 roles | 0 | 0 | Both FAILs are harness issues (see §3) |

The 2 skips are pre-existing and unrelated to this pass.

## 3. Features

| Feature | Tracker ID | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| Shared async client + timeouts | A1.1 | VERIFIED LIVE | `src/llm/runtime.py`; live analyze calls succeeded | — |
| Async streaming | A1.2 | VERIFIED LIVE | `refinement_delta`/`translation_delta` events in live AR/EN conversation | — |
| Retry with jitter + Retry-After | A1.3 | VERIFIED WITH MOCKS | `tests/test_concurrency.py`; `src/llm/retry.py` | Open decision: retry on connection errors |
| Per-model concurrency limits | A1.4 | VERIFIED WITH MOCKS | `tests/test_concurrency.py`; `src/llm/limiter.py` | — |
| 30-parallel test + loop-lag check | A1.5 | VERIFIED WITH MOCKS | `test_concurrency_30_parallel_analyses` | — |
| DeepSeek refinement + translation | — | VERIFIED LIVE | Live AR doctor/conversation; smoke: DeepSeek 20/20 PASS ×2 | — |
| Luna extraction + gpt-4.1-mini fallback | — | VERIFIED LIVE (Luna) / MOCKS (fallback) | Live `extraction: gpt-5.6-luna`; smoke gpt-4.1-mini 20/20 | Fallback path not triggered live |
| Reasoning routing + tier fallbacks (router OFF) | — | VERIFIED LIVE (simple) / MOCKS (complex) | Live `reasoning: gpt-5.6-luna`; smoke Sol/Terra 20/20 | Router OFF by default |
| Reasoning-pending behavior | — | VERIFIED WITH MOCKS | `tests/test_analyze_streaming.py` | — |
| Doctor-selected model fields | — | VERIFIED LIVE | `GET /models` lists Luna+Sol; `extraction_model`/`reasoning_model` accepted | — |
| `/health` and `/health/live` | A7.1.1 | VERIFIED LIVE | Live 200; `status: ok`; compose healthcheck uses `/health/live` | — |
| Telemetry without clinical text | A2.4 | VERIFIED LIVE | `usage.jsonl` has `transcript_hash`, no text; `phi_safe=true` | — |
| Quotas config | A3.2 | MANUAL | `docs/foundry_models.md` quota table | Fill TPM/RPM from portal |
| Smoke script | A3.3 | VERIFIED LIVE (partial) | Full run completed; 7/9 roles PASS, 0 content-filter blocks | 2 harness FAILs |
| Locust script + per-session CSV | A5 | NOT VERIFIED | `locustfile.py` | Not run (load test out of scope) |
| Capacity runbook | A6 | MANUAL | `docs/capacity_report_template.md` | — |
| CI pipeline | A4.1 | VERIFIED WITH MOCKS | `azure-pipelines.yml`; local CI simulation green | Needs Azure DevOps run |
| Staging + production approval files | A4.2/A4.3 | MANUAL | `pipelines/deploy-template.yml`; `docs/MANUAL_STEPS_AR.md` | Approvals configured in UI |
| Docker Compose run | — | VERIFIED LIVE | `docker compose up -d`; both healthy; endpoints 200 | — |
| Key Vault code path | A2.1 | VERIFIED WITH MOCKS | `src/core/azure_identity.py` | Live vault not provisioned |
| Entra-only check script | A2.5.2 | NOT VERIFIED | `scripts/check_entra_only_auth.py` | Needs Entra-only Foundry |

## 4. Changes

### Files changed (7 modified, 1 added vs the baseline manifest)

| File | Change |
| --- | --- |
| `Dockerfile` | Added `libsndfile1` (soundfile/librosa backend). |
| `docker-compose.yml` | Services `medvoice-app` + `medvoice-mongo`; project `medvoice`; `7777:7777`; `env_file`/`image` parameterized; liveness healthcheck; Mongo not host-published. |
| `azure-pipelines.yml` | CI runs 3 provider configs in a clean env; deploy stages add `MEDVOICE_COMPOSE_DIR`. |
| `pipelines/deploy-template.yml` | Rewritten to deploy the same Compose stack with healthcheck + smoke test + rollback. |
| `docs/cicd.md` | Documented the Compose-based deployment. |
| `docs/MANUAL_STEPS_AR.md` | Added the "تشغيل المشروع بالدوكر كومبوس" section. |
| `docs/foundry_models.md` | Added the smoke-script run results and interpretation. |
| `docs/work_log.md` | Added the finishing-pass section. |
| `src/requirements.txt` | Added `openpyxl==3.1.5` (startup import, was undeclared). |
| `pyproject.toml` | Added `openpyxl==3.1.5`. |
| `docs/FINAL_REPORT.md` | Added (this file). |

> The manifest diff reports `docs/MANUAL_STEPS_AR.md` as "added" because it was
> created after the baseline snapshot (birth 16:42:48 vs snapshot 16:38:09). It
> is a documentation file, not a code change.

### Files untouched (teammates' areas)

- `src/providers/**`, `src/core/admission.py`, gateway/Entra/APIM, validation layer (M3),
  ICD-10 lookup, dashboard auth, STT (`src/model/speech_service.py`), patient history
  (`src/model/patient_history.py`) — **no behavior changes**.
- `src/llm/retry.py`, `src/llm/runtime.py`, `src/llm/limiter.py` — **not changed in this pass**
  (the three one-line fixes from the previous pass remain; see the work log §7).

### One-line fixes in teammates' code (from the previous pass, unchanged here)

1. `src/llm/retry.py` — stop retrying `APIConnectionError`; cap attempts at 4 (restores A1.3.1).
2. `src/llm/runtime.py` — 429 no longer trips the circuit breaker; add `provider` property (restores A1.5.1 + M2 contract).
3. `src/llm/limiter.py` — distinguish an explicit `default_limit` from `LLM_LIMIT_DEFAULT`.

### Assumptions and decisions applied

- **Mongo in Compose**: added a `medvoice-mongo` service (dev-only, not host-published) and
  overrode only `MONGODB_URI` in the compose `environment` section. Safest option; the app
  reaches it over the compose network.
- **Healthcheck**: uses `/health/live` (liveness only) so an STT outage never restarts the container.
- **Deploy**: the pipeline runs `docker compose up -d --no-build` with the server's existing env
  file; it never builds on the server and never prints the env file.
- **`openpyxl`**: added to both dependency files (hard startup import via `evaluation/__init__.py`).

## 5. Open decisions (not changed by this pass)

1. Retry on connection errors (currently not retried).
2. Whether 429 counts in the circuit breaker (currently it does not).
3. `LLM_TOTAL_TIMEOUT_S=300`.
4. MongoDB credentials — local Compose uses an unauthenticated dev Mongo; production needs auth.
5. Clinical Lead sign-off on `src/prompts/routing_criteria.txt`.
6. Data residency — all deployments Global Standard.
7. Provider-abstraction overlap between `src/providers` and the foundry branch in `src/llm/runtime.py`.

## 6. What a human still has to do (in order)

Reference: `docs/MANUAL_STEPS_AR.md`.

1. Run the pipeline on Azure DevOps (CI → BuildPush → DeployDev; staging/production per parameters).
2. Fill the TPM/RPM quota table in `docs/foundry_models.md` from the Azure portal.
3. Get Clinical Lead sign-off on `src/prompts/routing_criteria.txt`.
4. Provision Key Vault + Managed Identity and enable `USE_KEY_VAULT` (portal work).
5. Configure Foundry networking and Entra-only auth, then run `scripts/check_entra_only_auth.py`.
6. Fix the smoke-script harness (Luna strict-JSON expectation; router prompt must contain "json").
7. (Optional) Run the Locust load test.

### Rollback

- Provider: set `LLM_PROVIDER=openrouter`.
- Routing: `ROUTER_ENABLED=false` (default).
- Compose: `docker compose down` then `python app.py` (from `src/`).
- Full source: restore from `/home/ai/Workspace/MedVoice_backup_20261021_163647`.

## 7. Exact commands

```bash
# start / stop / logs / health
docker compose up -d --build
docker compose down
docker compose logs -f medvoice-app
curl -s http://localhost:7777/health/live
curl -s http://localhost:7777/health
```
