# M2.4: Testing, CI Gates & Observability

## 1. Automated Test Suite Matrix
The M2 test suite satisfies the engineering tracker's **Definition of Done (DoD)** and is executed as a CI gate:

```bash
PYTHONPATH=src pytest tests/test_m2_*.py -v
```

| Test File | Test Cases | Objective |
|---|---|---|
| [`tests/test_m2_contracts.py`](file:///home/ai/Workspace/Doha/MedVoice/tests/test_m2_contracts.py) | 5 tests | Enforces that ABCs cannot be instantiated directly and that all abstract methods are present. Tests `ProviderHealthResult` immutability. |
| [`tests/test_m2_adapters.py`](file:///home/ai/Workspace/Doha/MedVoice/tests/test_m2_adapters.py) | 11 tests | Verifies that adapters delegate faithfully without altering signatures. Tests async stream consumption and probe error handling. |
| [`tests/test_m2_factory.py`](file:///home/ai/Workspace/Doha/MedVoice/tests/test_m2_factory.py) | 8 tests | Verifies default selection, unknown engine rejection (`ValueError`), singleton identity, and lifespan teardown. |
| [`tests/test_m2_switch.py`](file:///home/ai/Workspace/Doha/MedVoice/tests/test_m2_switch.py) | 3 tests | Tests end-to-end switching between providers using mocked environments, and validates the `/health` endpoint probe. |

---

## 2. Health Endpoint & Observability (`/health`)

The health monitoring probe in `src/app.py` has been updated to query through the provider abstraction:

```python
async def _check_llm(request: Request) -> dict:
    from providers.factory import get_llm
    provider = getattr(request.app.state, "llm_provider", None) or get_llm()
    try:
        result = await provider.health_check()
        return {
            "component": "llm",
            "provider": getattr(provider, "provider_name", os.getenv("LLM_PROVIDER", "foundry")),
            "status": "ok" if result.ok else "degraded",
            "latency_ms": result.latency_ms,
            "detail": result.detail,
        }
    except Exception as exc:
        return {
            "component": "llm",
            "provider": getattr(provider, "provider_name", os.getenv("LLM_PROVIDER", "foundry")),
            "status": "error",
            "detail": str(exc),
        }
```

### Response Schema on `/health`:
```json
{
  "status": "ok",
  "checks": {
    "llm": {
      "component": "llm",
      "provider": "foundry",
      "status": "ok",
      "latency_ms": 42.15,
      "detail": null
    },
    "mongodb": {
      "component": "mongodb",
      "status": "ok",
      "latency_ms": 2.11
    }
  }
}
```

### Observability Benefits:
1. **Zero Token Cost**: Probing uses lightweight metadata endpoints (e.g. `models.list()`) rather than generating completion tokens.
2. **Provider Transparency**: DevOps dashboards and Prometheus scrapers can monitor which provider is active and detect latency spikes before traffic is impacted.
3. **Graceful Degradation**: Probe failures return `status: "degraded"` or `"error"` with error details without crashing the FastAPI application.
