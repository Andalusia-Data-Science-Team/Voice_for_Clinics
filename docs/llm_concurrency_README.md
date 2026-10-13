# LLM concurrency runtime

The FastAPI application owns one `AsyncOpenAI` client and one HTTP connection
pool per worker. The client is created in the application lifespan and closed
on shutdown. LLM calls do not construct clients per request and streaming uses
native asynchronous iteration.

## Configuration

Set these environment variables before starting the application:

| Variable | Default | Description |
| --- | ---: | --- |
| `AI_FOUNDRY_ENDPOINT` | — | Azure AI Foundry endpoint URL |
| `AI_FOUNDRY_KEY` | — | Primary API key for Azure AI Foundry |
| `AI_FOUNDRY_API_VERSION` | `2024-06-01` | Azure AI Foundry API version |
| `AI_FOUNDRY_AUTH_METHOD` | `api_key` | Authentication method (`api_key` or `managed_identity`) |
| `LLM_TIMEOUT_S` | `45` | Per-attempt SDK request timeout |
| `LLM_TOTAL_TIMEOUT_S` | `300` | Overall retry deadline, including attempts and sleeps |
| `LLM_CONNECT_TIMEOUT_S` | `5` | Connection timeout |
| `LLM_MAX_ATTEMPTS` | `4` | Total attempts, including the first request |
| `LLM_BACKOFF_MAX_S` | `8` | Maximum exponential-jitter backoff |
| `LLM_RETRY_AFTER_MAX_S` | `30` | Maximum accepted provider Retry-After delay |
| `LLM_LIMIT_DEFAULT` | `8` | Per-model semaphore fallback |
| `LLM_MAX_CONNECTIONS` | `100` | HTTP connection-pool maximum |
| `LLM_MAX_KEEPALIVE_CONNECTIONS` | `20` | HTTP keep-alive pool maximum |
| `LLM_LIMIT__<MODEL>` | — | Model-specific limit; non-alphanumeric model characters become `_` |

For example, `LLM_LIMIT__DEEPSEEK_V3_2=6` overrides the limit
for the DeepSeek-V3.2 refinement/translation deployment.

Limits are process-local. With multiple Uvicorn workers or containers, each
worker has its own client, pool, and semaphores. Size a limit using Little's
law: `in_flight ≈ calls/s × latency`; for example, 30 encounters/minute,
two calls each, and eight seconds of latency imply approximately eight slots.

## Retry behavior

The retry policy uses full-jitter exponential backoff. For retry number `r`
(starting at 1), the delay is
`uniform(0, min(LLM_BACKOFF_MAX_S, 1 × 2^(r-1)))` seconds: a 1-second base,
multiplier 2, and default 8-second cap. When `Retry-After` is present, the
sleep is `max(jitter, capped Retry-After)`; the Retry-After cap defaults to
30 seconds and is configured independently of the backoff cap.

`LLM_TIMEOUT_S` limits each SDK request attempt. `LLM_TOTAL_TIMEOUT_S` bounds
the complete non-stream operation, including retries and backoff sleeps. Its
300-second default exceeds the 270-second modeled worst case
(`4 × 45s + 3 × 30s`) for four request attempts and three maximum
`Retry-After` sleeps. For streams, the total deadline applies until the first
content chunk is emitted; the stream can continue after that chunk without
the initial-output deadline.

The retry policy retries only:

- HTTP 429;
- HTTP 5xx;
- OpenAI timeout errors; and
- HTTPX timeout errors.

HTTP 400, 401, 403, 404, 422, malformed model output, and
`insufficient_quota` are not retried. A provider
`Retry-After` value in seconds or HTTP-date form is honored up to
`LLM_RETRY_AFTER_MAX_S`. Azure AI Foundry authentication headers are configured
on the shared async client.

Streaming holds its model slot while requesting or consuming a stream and
releases it before a retry backoff sleep. A stream can be retried only before
the first content chunk is yielded; after output is emitted, the error is
surfaced to avoid duplicated text. Logs include model, attempt, status, wait,
and in-flight metrics only. Prompts, response bodies, API keys, and PHI are not
logged by the runtime.

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
| 30 parallel analyses; 429 then 200; max event-loop lag under 50 ms | `test_concurrency_30_parallel_analyses` | passing |
| Persistent mocked 429 limited to four attempts | `test_concurrency_mocked_429_has_at_most_three_retries` | passing |
| Retry-After honored and capped | `test_concurrency_retry_after_honoured` | passing |
| Non-retryable 400/401/403/404/422 | `test_concurrency_no_retry_on_4xx` | passing |
| Insufficient quota raises without retry | `test_concurrency_insufficient_quota_not_retried` | passing |
| Four-attempt maximum | `test_concurrency_max_attempts_is_four` | passing |
| 5xx and timeout retry | `test_concurrency_5xx_and_timeout_retried` | passing |
| Semaphore cap | `test_concurrency_semaphore_cap` | passing |
| Per-model isolation | `test_concurrency_per_model_isolation` | passing |
| Release after error and cancellation | `test_concurrency_semaphore_released_on_error_and_cancel` | passing |
| Ten async streams do not block the loop | `test_concurrency_stream_no_event_loop_block` | passing |
| Stream retry only before first chunk | `test_concurrency_stream_retry_only_before_first_chunk` | passing |
| Stream attempts never exceed four | `test_concurrency_stream_attempt_count_never_exceeds_four` | passing |
| Stream slot released during backoff | `test_concurrency_stream_releases_slot_during_backoff` | passing |
| Stream close error still releases slot | `test_concurrency_stream_aclose_error_releases_slot` | passing |
| Overall retry deadline | `test_concurrency_overall_deadline_stops_retry_loop` | passing |
| Stream deadline ends at first content | `test_concurrency_stream_deadline_stops_at_first_content` | passing |
| One shared client construction | `test_concurrency_single_client_instance` | passing |
| Model-specific environment override | `test_concurrency_config_limits_override` | passing |

Additional focused tests cover retry classification and cancellation of an
active stream: `test_concurrency_retry_classification`,
`test_concurrency_retry_transport_errors_and_quota_body`, and
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

Run it only when a live evaluation is intended:

```text
RUN_REAL_USE_CASE_EVAL=1 \
	.venv/bin/python -m pytest tests/test_real_use_case_models.py -s
```

Useful overrides:

- `REAL_USE_CASE_XLSX=/path/to/real_use_cases.xlsx`
- `REAL_USE_CASE_REPORT=/path/to/report.xlsx`
- `REAL_USE_CASE_FAIL_ON_ERROR=1` to fail on any provider/model error.

The six-scenario report is a live-provider test and is intentionally excluded
from `pytest -k concurrency`. Each detailed result now includes a 0–100 judge
score, a short justification, and the judge model used. OpenAI candidates are
judged by Claude Opus 5.5; all other candidates are judged by GPT 5.6 Sol.
Override those judge slugs with `JUDGE_SLUG_FOR_OPENAI` and
`JUDGE_SLUG_FOR_NON_OPENAI` when needed.
