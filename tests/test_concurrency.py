"""Deterministic tests for shared LLM runtime concurrency behavior."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from openai import AsyncOpenAI as RealAsyncOpenAI

import llm.retry as retry_module
import llm.runtime as runtime_module
from core.config import Config
from llm.limiter import ModelLimiter
from llm.retry import is_insufficient_quota, is_retryable, retry_delay, run_with_retries


class StatusError(Exception):
    """Minimal provider-like exception for retry tests."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


def _completion(text: str = "ok", model: str = "test-model") -> dict:
    """Return a minimal OpenAI-compatible completion payload."""
    return {
        "id": "completion-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def _sse_chunk(text: str, model: str = "test-model") -> bytes:
    """Encode one minimal chat-completion streaming chunk."""
    payload = {
        "id": "stream-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class _SSEStream(httpx.AsyncByteStream):
    """Async byte stream used to exercise the SDK's SSE parser."""

    def __init__(self, chunks: list[bytes], failure: BaseException | None = None,
                 gate: asyncio.Event | None = None,
                 started: asyncio.Event | None = None) -> None:
        self._chunks = chunks
        self._failure = failure
        self._gate = gate
        self._started = started

    async def __aiter__(self):
        if self._started is not None:
            self._started.set()
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(0)
        if self._gate is not None:
            await self._gate.wait()
        if self._failure is not None:
            raise self._failure
        yield b"data: [DONE]\n\n"


def _patch_runtime_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
    keys: list[str] | None = None,
):
    """Construct one real AsyncOpenAI client over an in-memory transport for Azure AI Foundry."""
    transport = httpx.MockTransport(handler)
    original = RealAsyncOpenAI

    def factory(*args, **kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=transport)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "AsyncOpenAI", factory)
    monkeypatch.setenv("AI_FOUNDRY_KEY", "primary-key")
    monkeypatch.setattr(Config, "AI_FOUNDRY_KEY", "primary-key")
    monkeypatch.setenv("AI_FOUNDRY_ENDPOINT", "https://mock-foundry.openai.azure.com/openai/v1")
    monkeypatch.setattr(Config, "AI_FOUNDRY_ENDPOINT", "https://mock-foundry.openai.azure.com/openai/v1")
    monkeypatch.setenv("AI_FOUNDRY_AUTH_METHOD", "api_key")
    monkeypatch.setattr(Config, "AI_FOUNDRY_AUTH_METHOD", "api_key")
    return runtime_module.LLMRuntime()


def _authorization(key: str) -> str:
    return "Bearer " + key


async def _no_wait(_: float) -> None:
    """Avoid real retry delays in deterministic tests."""


async def _collect_stream(runtime, model: str = "test-model") -> str:
    """Consume a runtime stream and return its concatenated content."""
    chunks = []
    async for chunk in runtime.stream(
        model=model,
        create_kwargs={"model": model, "messages": [{"role": "user", "content": "stream"}]},
    ):
        chunks.append(chunk)
    return "".join(chunks)


@pytest.mark.asyncio
async def test_concurrency_retryable_429_reaches_success():
    """A transient 429 is retried and then succeeds."""
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise StatusError(429)
        return "ok"

    result = await run_with_retries(operation, model="test-model", sleep=_no_wait)
    assert result == "ok"
    assert attempts == 2


@pytest.mark.asyncio
async def test_concurrency_foundry_headers_applied(monkeypatch):
    """Azure AI Foundry requests include api-key or api-version headers."""
    api_versions: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        api_versions.append(request.headers.get("api-version"))
        return httpx.Response(200, json=_completion())

    monkeypatch.setenv("AI_FOUNDRY_API_VERSION", "2024-05-01-preview")
    monkeypatch.setattr(Config, "AI_FOUNDRY_API_VERSION", "2024-05-01-preview")
    runtime = _patch_runtime_client(monkeypatch, handler)
    try:
        result = await runtime.complete(
            model="DeepSeek-V3.2",
            create_kwargs={
                "model": "DeepSeek-V3.2",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert result.choices[0].message.content == "ok"
        assert api_versions == ["2024-05-01-preview"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_no_retry_on_4xx():
    """Client errors are surfaced without retry."""
    for status in (400, 401, 403, 404, 422):
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            raise StatusError(status)

        with pytest.raises(StatusError):
            await run_with_retries(operation, model="test-model", sleep=_no_wait)
        assert attempts == 1


@pytest.mark.asyncio
async def test_concurrency_max_attempts_is_four(monkeypatch):
    """The retry policy never exceeds four attempts even if configured higher."""
    attempts = 0
    monkeypatch.setattr(retry_module.Config, "LLM_MAX_ATTEMPTS", 10)

    async def operation():
        nonlocal attempts
        attempts += 1
        raise StatusError(429)

    with pytest.raises(StatusError):
        await run_with_retries(operation, model="test-model", sleep=_no_wait)
    assert attempts == 4


@pytest.mark.asyncio
async def test_concurrency_semaphore_cap():
    """The model limiter never exceeds its configured cap."""
    limiter = ModelLimiter(default_limit=3)
    active = 0
    maximum = 0
    guard = asyncio.Lock()

    async def call():
        nonlocal active, maximum
        await limiter.acquire("model-a")
        async with guard:
            active += 1
            maximum = max(maximum, active)
        await asyncio.sleep(0)
        async with guard:
            active -= 1
        await limiter.release("model-a")

    await asyncio.gather(*(call() for _ in range(20)))
    assert maximum == 3


@pytest.mark.asyncio
async def test_concurrency_per_model_isolation():
    """A full model-A limiter does not block model B."""
    limiter = ModelLimiter(default_limit=1)
    await limiter.acquire("model-a")
    completed = asyncio.Event()

    async def model_b():
        await limiter.acquire("model-b")
        completed.set()
        await limiter.release("model-b")

    task = asyncio.create_task(model_b())
    await asyncio.wait_for(completed.wait(), timeout=0.2)
    await task
    await limiter.release("model-a")


@pytest.mark.asyncio
async def test_concurrency_semaphore_released_on_error_and_cancel():
    """A caller's finally block can safely return slots after failures/cancel."""
    limiter = ModelLimiter(default_limit=1)

    async def failing():
        await limiter.acquire("model-a")
        try:
            raise RuntimeError("failure")
        finally:
            await limiter.release("model-a")

    with pytest.raises(RuntimeError):
        await failing()

    async def blocked():
        await limiter.acquire("model-a")
        try:
            await asyncio.sleep(10)
        finally:
            await limiter.release("model-a")

    task = asyncio.create_task(blocked())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await limiter.acquire("model-a")
    await limiter.release("model-a")


def test_concurrency_retry_classification():
    """Only throttling and server failures are retryable among HTTP errors."""
    assert is_retryable(StatusError(429))
    assert is_retryable(StatusError(503))
    assert not is_retryable(StatusError(400))
    assert not is_retryable(StatusError(401))
    assert not is_retryable(StatusError(403))
    assert not is_retryable(StatusError(404))
    assert not is_retryable(StatusError(422))


def test_concurrency_retry_transport_errors_and_quota_body(monkeypatch):
    """Only timeout errors are retried among transport failures."""
    request = httpx.Request("POST", "https://example.test")
    assert not is_retryable(httpx.ConnectError("connect", request=request))
    assert not is_retryable(httpx.ReadError("read", request=request))
    assert not is_retryable(httpx.RemoteProtocolError("protocol", request=request))
    assert not is_retryable(openai.APIConnectionError(request=request))
    assert is_retryable(openai.APITimeoutError(request=request))
    assert is_retryable(httpx.TimeoutException("timeout", request=request))

    message_only = RuntimeError("billing issue, please contact support")
    message_only.body = {"error": {"message": "billing issue"}}
    assert not is_insufficient_quota(message_only)

    quota_error = RuntimeError("provider rejected request")
    quota_error.body = {"error": {"type": "insufficient_quota"}}
    assert is_insufficient_quota(quota_error)

    monkeypatch.setattr(retry_module.random, "uniform", lambda low, high: high)
    monkeypatch.setattr(retry_module.Config, "LLM_BACKOFF_MAX_S", 8.0)
    assert retry_delay(1, StatusError(429)) == 1.0
    assert retry_delay(2, StatusError(429)) == 2.0
    assert retry_delay(4, StatusError(429)) == 8.0


@pytest.mark.asyncio
async def test_concurrency_30_parallel_analyses(monkeypatch):
    """Thirty parallel analyses recover without event-loop scheduling stalls."""
    attempts = defaultdict(int)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_id = body["messages"][0]["content"]
        attempts[request_id] += 1
        if request_id == "warm-up":
            return httpx.Response(200, json=_completion(request_id, body["model"]))
        if attempts[request_id] == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=_completion(request_id, body["model"]))

    runtime = _patch_runtime_client(monkeypatch, handler)
    await runtime.complete(
        model="test-model",
        create_kwargs={
            "model": "test-model",
            "messages": [{"role": "user", "content": "warm-up"}],
        },
    )
    loop_lags: list[float] = []
    stop_sampler = asyncio.Event()
    sampler_task = None

    async def sample_loop_lag():
        loop = asyncio.get_running_loop()
        while not stop_sampler.is_set():
            scheduled = loop.time() + 0.01
            await asyncio.sleep(0.01)
            loop_lags.append(max(0.0, loop.time() - scheduled))

    try:
        sampler_task = asyncio.create_task(sample_loop_lag())

        async def call(index: int):
            request_id = f"request-{index}"
            result = await runtime.complete(
                model="test-model",
                create_kwargs={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": request_id}],
                },
            )
            return result.choices[0].message.content

        results = await asyncio.gather(*(call(index) for index in range(30)))
        stop_sampler.set()
        await sampler_task
        assert results == [f"request-{index}" for index in range(30)]
        request_attempts = {
            request_id: count for request_id, count in attempts.items()
            if request_id != "warm-up"
        }
        assert len(request_attempts) == 30
        assert all(1 <= count <= 4 for count in request_attempts.values())
        assert max(loop_lags, default=0.0) < 0.05
    finally:
        stop_sampler.set()
        if sampler_task is not None:
            await sampler_task
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_mocked_429_has_at_most_three_retries(monkeypatch):
    """Four persistent mocked 429 responses mean one try and three retries."""
    calls = 0

    async def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            request=request,
            json={"error": {"message": "busy"}},
        )

    runtime = _patch_runtime_client(monkeypatch, handler)
    monkeypatch.setattr(Config, "LLM_MAX_ATTEMPTS", 10)

    async def no_wait(_: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    try:
        with pytest.raises(openai.RateLimitError):
            await runtime.complete(
                model="test-model",
                create_kwargs={"model": "test-model", "messages": []},
            )
        assert calls == 4
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_retry_after_honoured(monkeypatch):
    """Retry-After is honored and capped by the configured maximum."""
    responses = iter([
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, json=_completion()),
    ])

    async def handler(_: httpx.Request):
        return next(responses)

    runtime = _patch_runtime_client(monkeypatch, handler)
    captured: list[float] = []

    async def capture(delay: float):
        captured.append(delay)

    monkeypatch.setattr(retry_module.Config, "LLM_RETRY_AFTER_MAX_S", 30.0)
    monkeypatch.setattr(retry_module.Config, "LLM_BACKOFF_MAX_S", 0.0)
    try:
        await retry_module.run_with_retries(
            lambda: runtime.client.chat.completions.create(
                model="test-model", messages=[],
            ),
            model="test-model",
            sleep=capture,
        )
        assert captured and captured[0] >= 2.0

        capped = StatusError(429, {"Retry-After": "60"})
        monkeypatch.setattr(retry_module.Config, "LLM_RETRY_AFTER_MAX_S", 30.0)
        assert retry_module.retry_after_seconds(capped) == 30.0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_stream_no_event_loop_block(monkeypatch):
    """Ten async streams complete while the event loop remains responsive."""
    async def handler(_: httpx.Request) -> httpx.Response:
        stream = _SSEStream([_sse_chunk("a"), _sse_chunk("b"), _sse_chunk("c")])
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    runtime = _patch_runtime_client(monkeypatch, handler)
    # Warm the SDK/MockTransport path before measuring scheduling latency so
    # import/validation setup is not mistaken for event-loop blocking.
    assert await _collect_stream(runtime) == "abc"
    loop_lags: list[float] = []
    stop = asyncio.Event()

    async def sampler():
        while not stop.is_set():
            scheduled = asyncio.get_running_loop().time() + 0.01
            await asyncio.sleep(0.01)
            loop_lags.append(max(0.0, asyncio.get_running_loop().time() - scheduled))

    sampler_task = asyncio.create_task(sampler())
    try:
        results = await asyncio.gather(*(_collect_stream(runtime) for _ in range(10)))
        stop.set()
        await sampler_task
        assert results == ["abc"] * 10
        assert max(loop_lags, default=0.0) < 0.05
    finally:
        stop.set()
        sampler_task.cancel()
        await asyncio.gather(sampler_task, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_stream_retry_only_before_first_chunk(monkeypatch):
    """A pre-output failure retries, while post-output failure does not."""
    calls = 0

    async def before_handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"message": "temporary"}})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SSEStream([_sse_chunk("full")]),
        )

    runtime = _patch_runtime_client(monkeypatch, before_handler)
    monkeypatch.setattr(runtime_module.asyncio, "sleep", _no_wait)
    monkeypatch.setattr(runtime_module.Config, "LLM_BACKOFF_MAX_S", 0.0)
    try:
        assert await asyncio.wait_for(_collect_stream(runtime), timeout=2.0) == "full"
        assert calls == 2
    finally:
        await runtime.close()

    calls = 0
    gate = asyncio.Event()

    async def after_handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SSEStream([_sse_chunk("first")], StatusError(503)),
        )

    runtime = _patch_runtime_client(monkeypatch, after_handler)
    try:
        with pytest.raises(StatusError):
            await _collect_stream(runtime)
        assert calls == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_stream_attempt_count_never_exceeds_four(monkeypatch):
    """Even a higher configured limit cannot yield more than four stream tries."""
    runtime = _patch_runtime_client(monkeypatch, lambda _: httpx.Response(200))
    create = AsyncMock(side_effect=StatusError(503))
    monkeypatch.setattr(runtime.client.chat.completions, "create", create)
    monkeypatch.setattr(Config, "LLM_MAX_ATTEMPTS", 10)
    monkeypatch.setattr(Config, "LLM_BACKOFF_MAX_S", 0.0)

    async def no_wait(_: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    try:
        with pytest.raises(StatusError):
            await _collect_stream(runtime)
        assert create.await_count == 4
        assert (await runtime.limiter.stats("test-model")).in_flight == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_stream_releases_slot_during_backoff(monkeypatch):
    """A failed stream releases its semaphore before sleeping for a retry."""
    async def handler(_: httpx.Request):
        return httpx.Response(200, json=_completion())

    runtime = _patch_runtime_client(monkeypatch, handler)
    sleep_entered = asyncio.Event()
    resume = asyncio.Event()

    class ContentStream:
        def __aiter__(self):
            async def chunks():
                yield SimpleNamespace(choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="ok")),
                ])
            return chunks()

        async def aclose(self):
            return None

    create = AsyncMock(side_effect=[StatusError(503), ContentStream()])
    monkeypatch.setattr(runtime.client.chat.completions, "create", create)

    async def observe_sleep(_: float):
        stats = await runtime.limiter.stats("test-model")
        assert stats.in_flight == 0
        sleep_entered.set()
        await resume.wait()

    monkeypatch.setattr(runtime_module.asyncio, "sleep", observe_sleep)
    monkeypatch.setattr(runtime_module.Config, "LLM_BACKOFF_MAX_S", 0.0)
    task = asyncio.create_task(_collect_stream(runtime))
    try:
        await asyncio.wait_for(sleep_entered.wait(), timeout=1.0)
        assert (await runtime.limiter.stats("test-model")).in_flight == 0
        resume.set()
        assert await asyncio.wait_for(task, timeout=1.0) == "ok"
    finally:
        resume.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_stream_aclose_error_releases_slot(monkeypatch):
    """A failing stream close cannot skip semaphore release."""
    async def handler(_: httpx.Request):
        return httpx.Response(200, json=_completion())

    runtime = _patch_runtime_client(monkeypatch, handler)

    class FailingCloseStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StatusError(503)

        async def aclose(self):
            raise RuntimeError("close failed")

    monkeypatch.setattr(
        runtime.client.chat.completions,
        "create",
        AsyncMock(return_value=FailingCloseStream()),
    )
    try:
        with pytest.raises(RuntimeError, match="close failed"):
            await _collect_stream(runtime)
        assert (await runtime.limiter.stats("test-model")).in_flight == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_overall_deadline_stops_retry_loop(monkeypatch):
    """The total timeout bounds all attempts and retry sleeps."""
    calls = 0

    async def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, request=request)

    runtime = _patch_runtime_client(monkeypatch, handler)
    monkeypatch.setattr(Config, "LLM_TOTAL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(retry_module.Config, "LLM_TOTAL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(retry_module.Config, "LLM_BACKOFF_MAX_S", 8.0)
    monkeypatch.setattr(retry_module.random, "uniform", lambda low, high: high)
    try:
        with pytest.raises(TimeoutError):
            await runtime.complete(
                model="test-model",
                create_kwargs={"model": "test-model", "messages": []},
            )
        assert calls == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_stream_deadline_stops_at_first_content(monkeypatch):
    """The stream total deadline is disabled after its first content chunk."""
    async def handler(_: httpx.Request):
        return httpx.Response(200, json=_completion())

    runtime = _patch_runtime_client(monkeypatch, handler)
    first_chunk = asyncio.Event()
    continue_stream = asyncio.Event()

    class DelayedStream:
        def __aiter__(self):
            async def chunks():
                yield SimpleNamespace(choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="first")),
                ])
                await continue_stream.wait()
            return chunks()

        async def aclose(self):
            return None

    monkeypatch.setattr(Config, "LLM_TOTAL_TIMEOUT_S", 0.02)
    monkeypatch.setattr(
        runtime.client.chat.completions,
        "create",
        AsyncMock(return_value=DelayedStream()),
    )

    async def consume():
        async for chunk in runtime.stream(
            model="test-model",
            create_kwargs={"model": "test-model", "messages": []},
        ):
            assert chunk == "first"
            first_chunk.set()
            await continue_stream.wait()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(first_chunk.wait(), timeout=1.0)
        await asyncio.sleep(0.05)
        assert not task.done()
        continue_stream.set()
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        continue_stream.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_single_client_instance(monkeypatch):
    """A load run reuses one client configured with the runtime timeout and no SDK retries."""
    constructed = 0
    client_options = []

    async def handler(_: httpx.Request):
        return httpx.Response(200, json=_completion())

    transport = httpx.MockTransport(handler)
    original = RealAsyncOpenAI

    def factory(*args, **kwargs):
        nonlocal constructed
        constructed += 1
        client_options.append({
            "timeout": kwargs["timeout"],
            "max_retries": kwargs["max_retries"],
        })
        kwargs["http_client"] = httpx.AsyncClient(transport=transport)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "AsyncOpenAI", factory)
    monkeypatch.setattr(Config, "LLM_API_KEY", "primary-key")
    monkeypatch.setattr(Config, "LLM_TIMEOUT_S", 45.0)
    monkeypatch.setattr(Config, "LLM_CONNECT_TIMEOUT_S", 5.0)
    runtime = runtime_module.LLMRuntime()
    try:
        await asyncio.gather(*(
            runtime.complete(
                model="test-model",
                create_kwargs={"model": "test-model", "messages": []},
            )
            for _ in range(30)
        ))
        assert constructed == 1
        assert client_options[0]["timeout"].read == 45.0
        assert client_options[0]["timeout"].connect == 5.0
        assert client_options[0]["max_retries"] == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_config_limits_override(monkeypatch):
    """A model-specific environment setting overrides only that model."""
    monkeypatch.setenv("LLM_LIMIT__MODEL_A", "2")
    monkeypatch.setattr(Config, "LLM_LIMIT_DEFAULT", 5)
    limiter = ModelLimiter()
    await limiter.acquire("model-a")
    await limiter.acquire("model-a")
    stats = await limiter.stats("model-a")
    assert stats.in_flight == 2

    blocked = asyncio.create_task(limiter.acquire("model-a"))
    await asyncio.sleep(0)
    assert not blocked.done()
    await limiter.release("model-a")
    await blocked
    await limiter.release("model-a")
    await limiter.release("model-a")

    for _ in range(5):
        await limiter.acquire("model-b")
    stats_b = await limiter.stats("model-b")
    assert stats_b.in_flight == 5
    for _ in range(5):
        await limiter.release("model-b")


@pytest.mark.asyncio
async def test_concurrency_insufficient_quota_not_retried(monkeypatch):
    """An insufficient-quota error is surfaced after one request."""
    attempts = 0
    uses_primary_key = False

    async def handler(request: httpx.Request):
        nonlocal attempts, uses_primary_key
        attempts += 1
        uses_primary_key = request.headers.get("authorization") == _authorization("primary-key")
        return httpx.Response(
            402,
            json={"error": {"code": "insufficient_quota"}},
            request=request,
        )

    runtime = _patch_runtime_client(monkeypatch, handler)
    try:
        with pytest.raises(openai.APIStatusError):
            await runtime.complete(
                model="test-model",
                create_kwargs={"model": "test-model", "messages": []},
            )
        assert attempts == 1
        assert uses_primary_key
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_concurrency_5xx_and_timeout_retried(monkeypatch):
    """A 5xx and an SDK timeout each retry before succeeding."""
    responses = iter([httpx.Response(503), httpx.Response(200, json=_completion())])

    async def handler(_: httpx.Request):
        return next(responses)

    runtime = _patch_runtime_client(monkeypatch, handler)
    try:
        result = await runtime.complete(
            model="test-model", create_kwargs={"model": "test-model", "messages": []},
        )
        assert result.choices[0].message.content == "ok"
    finally:
        await runtime.close()

    timeout_attempts = 0
    async def timeout_operation():
        nonlocal timeout_attempts
        timeout_attempts += 1
        if timeout_attempts == 1:
            raise openai.APITimeoutError(request=httpx.Request("POST", "http://test"))
        return "ok"

    assert await run_with_retries(timeout_operation, model="test-model", sleep=_no_wait) == "ok"
    assert timeout_attempts == 2


@pytest.mark.asyncio
async def test_concurrency_stream_cancel_releases_slot(monkeypatch):
    """Cancelling a live stream returns its model semaphore slot."""
    gate = asyncio.Event()
    started = asyncio.Event()

    async def handler(_: httpx.Request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SSEStream([_sse_chunk("first")], gate=gate, started=started),
        )

    runtime = _patch_runtime_client(monkeypatch, handler)
    task = asyncio.create_task(_collect_stream(runtime))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stats = await runtime.limiter.stats("test-model")
    assert stats.in_flight == 0
    gate.set()
    await runtime.close()
