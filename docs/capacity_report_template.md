# Capacity report template (A6)

Fill this in from the per-step Locust reports produced by
`scripts/run_load_ramp.sh`. **No numbers are invented here** — leave cells blank
until the run is done.

Ramp: 1 → 5 → 10 → 20 → 50 concurrent sessions, 10 minutes per step, per language.

## Per-step results

### Arabic

| Users | p50 latency (ms) | p95 latency (ms) | 429 count | Error count | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | | | | | |
| 5 | | | | | |
| 10 | | | | | |
| 20 | | | | | |
| 50 | | | | | |

### English

| Users | p50 latency (ms) | p95 latency (ms) | 429 count | Error count | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | | | | | |
| 5 | | | | | |
| 10 | | | | | |
| 20 | | | | | |
| 50 | | | | | |

## Before/after limit changes (A6.3.1)

If p95 or errors exceed target, adjust `LLM_LIMIT__<MODEL>` (A1.4) or APIM limits
and re-run the failing step.

| Step (language/users) | Limit changed | Before p95 / errors | After p95 / errors |
| --- | --- | --- | --- |
| | | | |
| | | | |

## Max safe sessions (A6.3.2) — one page

- **Max safe concurrent sessions (Arabic):** <TO FILL>
- **Max safe concurrent sessions (English):** <TO FILL>
- **Binding constraint:** <TO FILL — per-model semaphore, APIM, or provider quota>
- **Recommended `LLM_LIMIT__<MODEL>` values:** see `docs/foundry_models.md`.
- **Assumptions:** 30 encounters/min baseline, 8 s latency, Global Standard
  deployments, per-worker process-local limits.