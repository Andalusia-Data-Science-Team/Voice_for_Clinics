"""Tests for GET /health dependency probing (A7.1.1). Mocks only."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.requests import Request

import app as app_module


def _request():
    return Request({
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/health",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 9999),
    })


def _healthy():
    return SimpleNamespace(ok=True, latency_ms=12.0, detail=None)


def _unhealthy():
    return SimpleNamespace(ok=False, latency_ms=None, detail="boom")


def _llm_provider(ok: bool = True):
    """Fake LLM provider exposing the shared runtime used by _probe_foundry."""
    complete = AsyncMock(return_value=SimpleNamespace())
    if not ok:
        complete.side_effect = RuntimeError("foundry down")
    return SimpleNamespace(runtime=SimpleNamespace(complete=complete))


@pytest.fixture(autouse=True)
def _clear_health_cache():
    app_module._health_cache["at"] = 0.0
    app_module._health_cache["payload"] = None
    yield
    app_module._health_cache["at"] = 0.0
    app_module._health_cache["payload"] = None


@pytest.mark.asyncio
async def test_health_returns_200_when_all_ok():
    llm = _llm_provider(ok=True)
    stt = SimpleNamespace(provider_name="munsit", health_check=AsyncMock(return_value=_healthy()))
    request = _request()
    request.scope["app"] = SimpleNamespace(state=SimpleNamespace(llm_provider=llm, stt_provider=stt))

    response = await app_module.health_check(request)

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["status"] == "ok"
    components = {c["component"]: c["status"] for c in payload["checks"]}
    assert components["foundry"] == "ok"
    assert components["munsit"] == "ok"


@pytest.mark.asyncio
async def test_health_returns_503_when_foundry_fails():
    llm = _llm_provider(ok=False)
    stt = SimpleNamespace(provider_name="munsit", health_check=AsyncMock(return_value=_healthy()))
    request = _request()
    request.scope["app"] = SimpleNamespace(state=SimpleNamespace(llm_provider=llm, stt_provider=stt))

    response = await app_module.health_check(request)

    assert response.status_code == 503
    payload = json.loads(response.body)
    assert payload["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_health_returns_200_when_stt_degraded():
    """A down STT provider is 'degraded' but keeps HTTP 200 (owner decision)."""
    llm = _llm_provider(ok=True)
    stt = SimpleNamespace(provider_name="munsit", health_check=AsyncMock(return_value=_unhealthy()))
    request = _request()
    request.scope["app"] = SimpleNamespace(state=SimpleNamespace(llm_provider=llm, stt_provider=stt))

    response = await app_module.health_check(request)

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["status"] == "degraded"
    components = {c["component"]: c["status"] for c in payload["checks"]}
    assert components["munsit"] == "degraded"


@pytest.mark.asyncio
async def test_health_is_cached_for_30s():
    llm = _llm_provider(ok=True)
    stt = SimpleNamespace(provider_name="munsit", health_check=AsyncMock(return_value=_healthy()))
    request = _request()
    request.scope["app"] = SimpleNamespace(state=SimpleNamespace(llm_provider=llm, stt_provider=stt))

    await app_module.health_check(request)
    await app_module.health_check(request)

    # Second call served from cache: the Foundry probe ran only once.
    assert llm.runtime.complete.await_count == 1


@pytest.mark.asyncio
async def test_health_response_has_no_secrets_or_endpoints():
    llm = _llm_provider(ok=True)
    stt = SimpleNamespace(provider_name="munsit", health_check=AsyncMock(return_value=_healthy()))
    request = _request()
    request.scope["app"] = SimpleNamespace(state=SimpleNamespace(llm_provider=llm, stt_provider=stt))

    response = await app_module.health_check(request)

    body = response.body.decode().lower()
    for forbidden in ("key", "token", "secret", "http://", "https://", "bearer"):
        assert forbidden not in body