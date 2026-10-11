# LLM concurrency runtime

The FastAPI application owns one `AsyncOpenAI` client and one HTTP connection
pool per worker. The client is created in the application lifespan and closed
on shutdown. LLM calls do not construct clients per request and streaming uses
native asynchronous iteration.

## Configuration

Set these environment variables before starting the application:

| Variable | Default | Description |
| --- | ---: | --- |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible API base URL |
| `LLM_API_KEY` | `OPENROUTER_API_KEY` | Primary API key |
| `LLM_TIMEOUT_S` | `45` | Overall SDK request timeout |
| `LLM_CONNECT_TIMEOUT_S` | `5` | Connection timeout |
| `LLM_TOTAL_TIMEOUT_S` | `300` | Overall timeout deadline wrapping retry loops (stream deadline until first chunk) |
| `LLM_MAX_ATTEMPTS` | `4` | Total attempts, including the first request |
| `LLM_BACKOFF_MAX_S` | `8` | Maximum exponential-jitter backoff |
| `LLM_RETRY_AFTER_MAX_S` | `30` | Maximum accepted provider Retry-After delay |
| `LLM_LIMIT_DEFAULT` | `8` | Per-model semaphore fallback |
| `LLM_MAX_CONNECTIONS` | `100` | HTTP connection-pool maximum |
| `LLM_MAX_KEEPALIVE_CONNECTIONS` | `20` | HTTP keep-alive pool maximum |
| `LLM_LIMIT__<MODEL>` | — | Model-specific limit; non-alphanumeric model characters become `_` |

For example, `LLM_LIMIT__DEEPSEEK_DEEPSEEK_CHAT_V3_0324=6` overrides the limit
for the refinement/translation model.

Limits are process-local. With multiple Uvicorn workers or containers, each
worker has its own client, pool, and semaphores. Size a limit using Little's
law: `in_flight ≈ calls/s × latency`; for example, 30 encounters/minute,
two calls each, and eight seconds of latency imply approximately eight slots.

## Retry behavior

The retry policy uses bounded exponential jitter and retries only:

- HTTP 429;
- HTTP 5xx;
- OpenAI timeout/connection errors; and
- HTTPX timeout errors.

HTTP 400, 401, 403, 404, 422, malformed model output, and `insufficient_quota`
are not retried. `insufficient_quota` is detected strictly from `error.code` or
`error.type` and raises immediately. Key rotation has been entirely eliminated
from the runtime. A provider `Retry-After` value in seconds or HTTP-date form is
honored up to `LLM_RETRY_AFTER_MAX_S`.

Streaming acquires a model slot for active requests. If a retryable failure occurs
before the first content chunk is yielded, the model slot is explicitly released
before the full-jitter backoff sleep and re-acquired on the next attempt (up to 4
attempts total). The slot is released in an isolated `finally` block even if
stream `aclose()` raises. `LLM_TOTAL_TIMEOUT_S` enforces a deadline wrapping the
retry loop in `complete()` and until the first content chunk in `stream()`. Logs
include model, attempt, status, wait, and in-flight metrics only. Prompts, response
bodies, API keys, and PHI are not logged by the runtime.

## Verification

Install dependencies and run the deterministic suite:

```text
pytest -k concurrency -q
```

The tests use local mocks and do not call an external LLM provider. The legacy
live evaluation scripts under `src/` still require credentials and external
fixtures and are not part of this focused suite.

## Concurrency test matrix

Run the requested suite with:

```text
.venv/bin/python -m pytest -k concurrency -v
```

| Required case | Test name | Status |
| --- | --- | --- |
| 30 parallel analyses; 429 then 200 (event loop lag < 50ms) | `test_concurrency_30_parallel_analyses` | passing |
| Retry-After honored and capped | `test_concurrency_retry_after_honoured` | passing |
| Non-retryable 400/401/404/422 | `test_concurrency_no_retry_on_4xx` | passing |
| Insufficient quota; immediate failure | `test_concurrency_insufficient_quota_not_retried` | passing |
| Four-attempt maximum | `test_concurrency_max_attempts_is_four` | passing |
| 5xx and timeout retry | `test_concurrency_5xx_and_timeout_retried` | passing |
| Semaphore cap | `test_concurrency_semaphore_cap` | passing |
| Per-model isolation | `test_concurrency_per_model_isolation` | passing |
| Release after error and cancellation | `test_concurrency_semaphore_released_on_error_and_cancel` | passing |
| Ten async streams do not block the loop | `test_concurrency_stream_no_event_loop_block` | passing |
| Stream retry only before first chunk | `test_concurrency_stream_retry_only_before_first_chunk` | passing |
| Stream slot released during backoff sleep | `test_concurrency_stream_slot_released_during_backoff` | passing |
| Total timeout wrapping completion | `test_concurrency_total_timeout_complete` | passing |
| Total timeout wrapping stream until first chunk | `test_concurrency_total_timeout_stream` | passing |
| Stream slot released on aclose() exception | `test_concurrency_stream_aclose_raises_slot_still_released` | passing |
| Stream persistent 503 exactly 4 attempts | `test_concurrency_stream_persistent_errors_exactly_four_attempts` | passing |
| Quota detection strictly code/type, not "billing" | `test_concurrency_is_insufficient_quota_classification` | passing |
| complete() persistent 429 exactly 4 calls | `test_concurrency_complete_persistent_429_exactly_four_calls` | passing |
| Stream backoff bounds and Retry-After capping | `test_concurrency_stream_backoff_bounds_and_retry_after` | passing |
| One shared client construction | `test_concurrency_single_client_instance` | passing |
| Model-specific environment override | `test_concurrency_config_limits_override` | passing |

Additional focused tests cover retry classification and cancellation of an
active stream: `test_concurrency_retry_classification` and
`test_concurrency_stream_cancel_releases_slot`.

## Six-scenario model comparison report

`tests/test_real_use_case_models.py` evaluates the six scenarios in
`src/data/real_use_cases.xlsx`—two ER, two OPD, and two Dental cases—against
every model in the extraction and reasoning registries.

The test writes `reports/real_use_case_model_evaluation.xlsx` with:

- **Detailed Results**: one row per scenario, stage, model, and field, showing
	expected value, raw model output, latency, provider-reported cost, and token counts;
- **Summary**: one row per scenario/stage/model with latency, cost, and errors;
- **Errors**: any model calls that failed while still allowing partial reports.

Run it only when a live OpenRouter evaluation is intended:

```text
RUN_REAL_USE_CASE_EVAL=1 \
	.venv/bin/python -m pytest tests/test_real_use_case_models.py -s
```

Useful overrides:

- `REAL_USE_CASE_XLSX=/path/to/real_use_cases.xlsx`
- `REAL_USE_CASE_REPORT=/path/to/report.xlsx`
- `REAL_USE_CASE_FAIL_ON_ERROR=1` to fail on any provider/model error.

The six-scenario report is a live-provider test and is intentionally excluded
from `pytest -k concurrency`.
