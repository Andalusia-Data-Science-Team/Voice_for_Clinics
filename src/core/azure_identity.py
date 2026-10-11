"""
Azure Managed Identity + Key Vault — Phase 0 Foundation
========================================================
Design goals
------------
1. **Import-timing safe** — this module is imported BEFORE any other
   ``core.*`` module.  ``bootstrap()`` must be called at the very top of
   ``app.py`` (before any other import) so that every ``os.getenv()`` call
   inside Config class-body statements sees the Key Vault values already
   in the environment.

2. **Concurrent secret loading** — all Key Vault secrets are fetched in
   parallel via ``concurrent.futures.ThreadPoolExecutor``.  With ~15 secrets
   and ~60-80 ms per round-trip this cuts startup overhead from ~1 s to
   ~120 ms on a healthy VNet connection.

3. **Full secret coverage** — covers every secret currently in .env:
   API keys, per-service aliases (speech/refine/…), DB credentials, Celery
   credentials, and forward slots for Phase 3 (AI Foundry, private STT).

4. **Structured health state** — ``get_vault_health()`` is consumed by the
   ``/health`` endpoint in Phase 4 to surface Key Vault reachability without
   exposing secret values.

5. **Graceful degradation** — any individual secret failure is logged and
   skipped; a total connection failure falls back silently to .env so the
   app still starts in dev / staging with ``USE_KEY_VAULT=false``.

6. **Zero third-party imports at module load** — Azure SDK imports are
   deferred inside functions so the module can be imported even when the SDK
   is not installed (pure .env mode).

Startup call sequence (app.py)
-------------------------------
    # ── MUST be the very first two lines of app.py, before any other import ──
    from core.azure_identity import bootstrap
    bootstrap()                        # loads KV → os.environ BEFORE Config
    from core.config import Config     # now sees KV values as if from .env
    ... rest of imports ...
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime control flags  (read once at module import — before bootstrap())
# These are safe to read from os.environ directly because they are NOT secrets;
# they are infrastructure coordinates that belong in .env even in production.
# ---------------------------------------------------------------------------
_USE_KEY_VAULT: bool = os.getenv("USE_KEY_VAULT", "false").lower() == "true"
_VAULT_URL: str = os.getenv("AZURE_KEY_VAULT_URL", "")
_MAX_WORKERS: int = int(os.getenv("KV_FETCH_WORKERS", "8"))  # parallel secret fetches

# ---------------------------------------------------------------------------
# Secret catalogue
# ---------------------------------------------------------------------------
# Each tuple: (key_vault_secret_name, os.environ_var_name)
# Add new secrets here — nowhere else.
_SECRET_CATALOGUE: tuple[tuple[str, str], ...] = (
    # ── LLM providers ──────────────────────────────────────────────────────
    ("openrouter-api-key",      "OPENROUTER_API_KEY"),
    ("openrouter-api-keys",     "OPENROUTER_API_KEYS"),
    ("mistral-api-key",         "MISTRAL_API_KEY"),
    ("groq-api-key",            "GROQ_API_KEY"),
    ("fireworks-api-key",       "FIREWORKS_API_KEY"),
    # ── Per-service aliases (config.py SPEECH_API_KEY etc.) ───────────────
    ("speech-api-key",          "speech"),
    ("refine-api-key",          "refine"),
    ("translation-api-key",     "translation"),
    ("extraction-api-key",      "extraction"),
    ("questions-api-key",       "questions"),
    # ── STT ────────────────────────────────────────────────────────────────
    ("munsit-api-key",          "MUNSIT_API_KEY"),
    ("private-stt-key",         "PRIVATE_STT_KEY"),          # Phase 3 slot
    # ── AI Foundry (Phase 3) ───────────────────────────────────────────────
    ("ai-foundry-key",          "AI_FOUNDRY_KEY"),           # Phase 3 slot
    # ── Database ───────────────────────────────────────────────────────────
    ("patient-db-uid",          "PATIENT_DB_UID"),
    ("patient-db-pwd",          "PATIENT_DB_PWD"),
    # ── Celery / broker ────────────────────────────────────────────────────
    ("celery-broker-url",       "CELERY_BROKER_URL"),
    ("celery-result-backend",   "CELERY_RESULT_BACKEND"),
    ("celery-flower-password",  "CELERY_FLOWER_PASSWORD"),
)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
class _VaultStatus(str, Enum):
    NOT_CONFIGURED = "not_configured"   # USE_KEY_VAULT=false
    OK             = "ok"               # connected, secrets loaded
    PARTIAL        = "partial"          # connected but some secrets failed
    FAILED         = "failed"           # could not connect at all


@dataclass
class _VaultHealth:
    status: _VaultStatus = _VaultStatus.NOT_CONFIGURED
    loaded: int = 0
    failed: int = 0
    skipped: int = 0       # secrets with empty values (not an error)
    latency_ms: float = 0.0
    failed_names: list[str] = field(default_factory=list)
    error: Optional[str] = None


_health = _VaultHealth()

# In-process secret cache: env-var-name → secret value
# Written once at startup, never mutated afterwards.
_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bootstrap() -> None:
    """
    **Must be called before any other import in app.py.**

    Loads Key Vault secrets into ``os.environ`` so that subsequent
    ``os.getenv()`` calls — including those in ``Config`` class body
    statements — see the vault values.

    If ``USE_KEY_VAULT=false`` (default) this is a no-op and the app
    behaves exactly as before (pure .env).
    """
    global _cache, _health

    if not _USE_KEY_VAULT:
        logger.info("[KeyVault] USE_KEY_VAULT=false — using .env secrets")
        _health = _VaultHealth(status=_VaultStatus.NOT_CONFIGURED)
        return

    if not _VAULT_URL:
        logger.warning(
            "[KeyVault] USE_KEY_VAULT=true but AZURE_KEY_VAULT_URL is not set "
            "— falling back to .env"
        )
        _health = _VaultHealth(
            status=_VaultStatus.FAILED,
            error="AZURE_KEY_VAULT_URL not set",
        )
        return

    t_start = time.perf_counter()
    loaded, failed, skipped = _fetch_all_secrets_parallel()
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Inject into os.environ (setdefault keeps any existing value — useful
    # if an operator has overridden a single secret via a K8s secret env var)
    for env_name, value in loaded.items():
        os.environ.setdefault(env_name, value)
        _cache[env_name] = value

    failed_names = list(failed.keys())
    if failed and not loaded:
        status = _VaultStatus.FAILED
    elif failed:
        status = _VaultStatus.PARTIAL
    else:
        status = _VaultStatus.OK

    _health = _VaultHealth(
        status=status,
        loaded=len(loaded),
        failed=len(failed),
        skipped=skipped,
        latency_ms=round(elapsed_ms, 1),
        failed_names=failed_names,
    )

    logger.info(
        "[KeyVault] bootstrap complete: loaded=%d failed=%d skipped=%d latency=%.0fms",
        len(loaded), len(failed), skipped, elapsed_ms,
    )
    if failed_names:
        logger.warning("[KeyVault] Failed secrets: %s", ", ".join(failed_names))


def get_secret(env_name: str) -> Optional[str]:
    """
    Retrieve a secret by its *environment variable name*.

    Resolution order:
      1. In-process Key Vault cache (populated at startup)
      2. ``os.environ`` (covers .env fallback and any operator overrides)

    This is the preferred accessor for code that needs a secret at
    *request time* (e.g. building a DB connection string on demand).
    For code that only runs at startup, reading ``Config.*`` attributes is
    fine because ``bootstrap()`` has already populated ``os.environ``.
    """
    return _cache.get(env_name) or os.getenv(env_name)


def get_vault_health() -> dict:
    """
    Return a safe (no secret values) health snapshot for the /health endpoint.
    """
    return {
        "status":       _health.status.value,
        "loaded":       _health.loaded,
        "failed":       _health.failed,
        "skipped":      _health.skipped,
        "latency_ms":   _health.latency_ms,
        "failed_names": _health.failed_names,
        "error":        _health.error,
    }


def is_vault_healthy() -> bool:
    """True if the vault is not in use (NOT_CONFIGURED) or loaded successfully."""
    return _health.status in (_VaultStatus.NOT_CONFIGURED, _VaultStatus.OK)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_client():
    """
    Build and return a ``SecretClient``.  Deferred import keeps the module
    importable without the Azure SDK installed.

    ``DefaultAzureCredential`` tries, in order:
      1. Environment variables (AZURE_CLIENT_ID + AZURE_CLIENT_SECRET + AZURE_TENANT_ID)
      2. Workload Identity (AKS)
      3. Managed Identity (App Service / Container Apps / VM)
      4. Azure CLI  ← dev-laptop fallback
      5. Azure PowerShell
      6. Interactive browser (last resort)
    """
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    credential = DefaultAzureCredential(
        # Suppress noisy "trying X credential" logs at WARNING level.
        # Set to logging.DEBUG to see the full chain.
        logging_enable=False,
    )
    return SecretClient(vault_url=_VAULT_URL, credential=credential)


def _fetch_one(client, vault_name: str, env_name: str) -> tuple[str, str | None, str | None]:
    """
    Fetch a single secret.  Returns (env_name, value_or_None, error_or_None).
    Runs inside a thread — all Azure SDK calls are blocking.
    """
    try:
        secret = client.get_secret(vault_name)
        value = secret.value
        if not value:
            logger.debug("[KeyVault] Secret '%s' exists but is empty", vault_name)
            return env_name, None, None   # skipped (empty)
        logger.info("[KeyVault] Loaded: %s → %s", vault_name, env_name)
        return env_name, value, None
    except Exception as exc:
        # ResourceNotFoundError is common for Phase 3 slots not yet created.
        err_type = type(exc).__name__
        logger.warning("[KeyVault] Failed '%s': %s: %s", vault_name, err_type, exc)
        return env_name, None, str(exc)


def _fetch_all_secrets_parallel() -> tuple[dict[str, str], dict[str, str], int]:
    """
    Fetch all catalogue secrets in parallel.

    Returns
    -------
    loaded  : {env_name: value}   — successfully fetched, non-empty secrets
    failed  : {env_name: error}   — secrets that raised an exception
    skipped : int                 — secrets that exist but are empty
    """
    loaded: dict[str, str] = {}
    failed: dict[str, str] = {}
    skipped = 0

    try:
        client = _build_client()
    except Exception as exc:
        logger.error("[KeyVault] Cannot build SecretClient: %s", exc)
        # Return everything as failed so health reflects the real state
        return {}, {env: str(exc) for _, env in _SECRET_CATALOGUE}, 0

    tasks = {(vault_name, env_name) for vault_name, env_name in _SECRET_CATALOGUE}

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="kv-fetch") as pool:
        future_map = {
            pool.submit(_fetch_one, client, vault_name, env_name): (vault_name, env_name)
            for vault_name, env_name in tasks
        }
        for future in as_completed(future_map):
            env_name, value, error = future.result()
            if error is not None:
                failed[env_name] = error
            elif value is None:
                skipped += 1
            else:
                loaded[env_name] = value

    return loaded, failed, skipped
