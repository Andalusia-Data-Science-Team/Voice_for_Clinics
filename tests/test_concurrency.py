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
from llm.retry import is_retryable, run_with_retries


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
    """Construct one real AsyncOpenAI client over an in-memory transport."""
    transport = httpx.MockTransport(handler)
    original = RealAsyncOpenAI

    def factory(*args, **kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=transport)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "AsyncOpenAI", factory)
    monkeypatch.setenv("LLM_API_KEY", "primary-key")
    monkeypatch.setenv(
        "OPENROUTER_API_KEYS",
        ",".join(keys or ["primary-key"]),
    )
    return runtime_module.LLMRuntime()


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
async def test_concurrency_no_retry_on_4xx():
    """Client errors are surfaced without retry."""
    for status in (400, 401, 404, 422):
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            raise StatusError(status)

        with pytest.raises(StatusError):
            await run_with_retries(operation, model="test-model", sleep=_no_wait)
        assert attempts == 1


@pytest.mark.asyncio
async def test_concurrency_max_attempts_is_four():
    """Persistent throttling stops at the configured four attempts."""
    attempts = 0

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
    assert not is_retryable(StatusError(404))
    assert not is_retryable(StatusError(422))


@pytest.mark.asyncio
async def test_concurrency_30_parallel_analyses(monkeypatch):
    """Thirty real runtime completions recover from one 429 each."""
    attempts = defaultdict(int)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_id = body["messages"][0]["content"]
        attempts[request_id] += 1
        if attempts[request_id] == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=_completion(request_id, body["model"]))

    runtime = _patch_runtime_client(monkeypatch, handler)
    try:
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
        assert results == [f"request-{index}" for index in range(30)]
        assert len(attempts) == 30
        assert all(1 <= count <= 4 for count in attempts.values())
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
async def test_concurrency_single_client_instance(monkeypatch):
    """A load run creates one shared AsyncOpenAI client."""
    constructed = 0

    async def handler(_: httpx.Request):
        return httpx.Response(200, json=_completion())

    transport = httpx.MockTransport(handler)
    original = RealAsyncOpenAI

    def factory(*args, **kwargs):
        nonlocal constructed
        constructed += 1
        kwargs["http_client"] = httpx.AsyncClient(transport=transport)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "AsyncOpenAI", factory)
    monkeypatch.setenv("LLM_API_KEY", "primary-key")
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
    """Quota failure skips retry and tries the next configured key once."""
    attempts: list[str] = []

    async def handler(request: httpx.Request):
        attempts.append(request.headers.get("authorization", "default"))
        if len(attempts) == 1:
            return httpx.Response(402, json={"error": {"code": "insufficient_quota"}})
        return httpx.Response(200, json=_completion("rotated"))

    runtime = _patch_runtime_client(
        monkeypatch,
        handler,
        keys=["primary-key", "secondary-key"],
    )
    try:
        result = await runtime.complete(
            model="test-model", create_kwargs={"model": "test-model", "messages": []},
        )
        assert result.choices[0].message.content == "rotated"
        assert len(attempts) == 2
        assert attempts[0] != attempts[1]
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
