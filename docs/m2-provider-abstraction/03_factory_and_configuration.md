# M2.3: Provider Factory, Configuration & Lifecycle

## 1. Overview & Zero-Code Switching
The Provider Factory (`src/providers/factory.py`) enables operators and developers to switch backends simply by editing environment variables in `.env` without modifying any application code.

### Supported Configuration Flags

```dotenv
# ── Provider switches (M2) ────────────────────────────────────
# LLM backend: foundry (default) | openrouter (future)
LLM_PROVIDER=foundry

# STT backend: munsit (Arabic default) | voxtral (English default)
STT_PROVIDER=munsit
```

---

## 2. Factory Registry Architecture (`src/providers/factory.py`)

### Factory Functions

| Function | Signature | Description |
|---|---|---|
| `build_llm_provider` | `(provider_name: Optional[str] = None) -> LLMProvider` | Builds a fresh instance of the configured LLM provider. Raises `ValueError` for unknown values. |
| `build_stt_provider` | `(provider_name: Optional[str] = None) -> STTProvider` | Builds a fresh instance of the configured STT provider. Automatically binds required API keys. |
| `get_llm` | `() -> LLMProvider` | Returns the process-wide `LLMProvider` singleton. Thread- and asyncio-safe. |
| `get_stt` | `() -> STTProvider` | Returns the process-wide `STTProvider` singleton. |
| `reset_providers` | `() -> None` | Resets stored singleton instances (useful in test teardown). |

---

## 3. Application Lifecycle (FastAPI `lifespan`)

FastAPI handles provider initialization and teardown inside the async lifespan context manager in `src/app.py`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from providers.factory import build_llm_provider, build_stt_provider

    # 1. Instantiate concrete providers once per worker process
    app.state.llm_provider = build_llm_provider()
    app.state.stt_provider = build_stt_provider()

    # 2. Backward compatibility alias for legacy code referencing app.state.llm
    app.state.llm = getattr(app.state.llm_provider, "runtime", app.state.llm_provider)

    try:
        app.state.db = await connect_db()
        app.state.sessions = SessionRepository(app.state.db)
        app.state.feedback_repo = FeedbackRepository(app.state.db)
        logger.info("[Lifespan] MongoDB connection and repositories initialized")
    except Exception as exc:
        ...

    try:
        yield
    finally:
        # 3. Graceful teardown on SIGTERM / SIGINT
        if hasattr(app.state.llm_provider, "close"):
            await app.state.llm_provider.close()
        elif hasattr(app.state.llm, "close"):
            await app.state.llm.close()
        await close_db()
```

### Why Backward Compatibility Matters:
During the transition phase, legacy modules might still access `request.app.state.llm`. By mapping `app.state.llm = getattr(app.state.llm_provider, "runtime", app.state.llm_provider)`, all legacy helpers (like `get_llm_runtime(request)`) continue to work seamlessly without breakage.

---

## 4. Package Import Mechanics (`src/providers/__init__.py`)

To prevent circular import issues between `providers.factory`, `providers.foundry_llm`, and `providers.base`, `src/providers/__init__.py` utilizes Python's module-level `__getattr__` hook:

```python
def __getattr__(name: str):
    """Lazy-load factory functions to avoid circular import issues."""
    if name in ("build_llm_provider", "build_stt_provider", "get_llm", "get_stt", "reset_providers"):
        from . import factory
        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

This guarantees that importing `from providers import LLMProvider, get_llm` works cleanly across the codebase.
