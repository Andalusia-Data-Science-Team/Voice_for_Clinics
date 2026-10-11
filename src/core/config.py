"""
Application configuration.
===========================
All secrets are read lazily via classmethods — never at class-body
evaluation time.  This means ``bootstrap()`` from ``core.azure_identity``
can safely be called BEFORE this module is imported, injecting Key Vault
values into ``os.environ``; but it also means that if ``Config.*`` is
accessed *after* ``bootstrap()``, the values are always fresh regardless
of import order.

Non-secret infrastructure coordinates (timeouts, limits, folder paths)
remain as class-level attributes because they are safe to bake in at
import time and are never stored in Key Vault.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Load .env from src/.env or root .env
_src_env = Path(__file__).resolve().parent.parent / ".env"
_root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _src_env.is_file():
    load_dotenv(_src_env)
elif _root_env.is_file():
    load_dotenv(_root_env)
else:
    load_dotenv()


class Config:
    """Base application configuration."""

    # ── Non-secret infrastructure ─────────────────────────────────────────
    MAX_CONTENT_LENGTH          = 16 * 1024 * 1024   # 16 MB max upload
    ALLOWED_EXTENSIONS          = {"mp3", "wav", "ogg"}
    DEBUG                       = True
    TESTING                     = False
    PORT                        = 8586
    UPLOAD_FOLDER               = os.getenv("UPLOAD_FOLDER", "uploads")
    DATABASE_PATH               = "app_data.db"

    PATIENT_HISTORY_CSV_PATH = os.getenv(
        "PATIENT_HISTORY_CSV_PATH",
        "/home/ai/Workspace/Doha/MedVoice/src/data/Visit Level History.csv",
    )
    XLSX_PATH = os.getenv(
        "XLSX_PATH",
        "/home/ai/Workspace/Doha/MedVoice/src/data/"
        "Medical_Voice_Assistant_Expected_Fields_Expanded.xlsx",
    )

    # ── LLM runtime tuning (not secrets) ─────────────────────────────────
    LLM_BASE_URL                  = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_TIMEOUT_S                 = float(os.getenv("LLM_TIMEOUT_S", "45"))
    LLM_CONNECT_TIMEOUT_S         = float(os.getenv("LLM_CONNECT_TIMEOUT_S", "5"))
    LLM_MAX_ATTEMPTS              = int(os.getenv("LLM_MAX_ATTEMPTS", "4"))
    LLM_BACKOFF_MAX_S             = float(os.getenv("LLM_BACKOFF_MAX_S", "8"))
    LLM_RETRY_AFTER_MAX_S         = float(os.getenv("LLM_RETRY_AFTER_MAX_S", "30"))
    LLM_LIMIT_DEFAULT             = int(os.getenv("LLM_LIMIT_DEFAULT", "8"))
    LLM_MAX_CONNECTIONS           = int(os.getenv("LLM_MAX_CONNECTIONS", "100"))
    LLM_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("LLM_MAX_KEEPALIVE_CONNECTIONS", "20"))

    # ── Database config (non-credential parts) ────────────────────────────
    PATIENT_DB_DRIVER = os.getenv("PATIENT_DB_DRIVER", "{ODBC Driver 17 for SQL Server}")
    BUSINESS_UNITS = {
        "AHJ": {"server": "10.111.111.169", "database": "AHJ.Env.HCMS"},
        "AMH": {"server": "172.31.44.169",  "database": "AMH.Env.HCMS"},
        "ANC": {"server": "172.31.44.169",  "database": "ANC.Env.HCMS"},
    }

    # ── Phase 3: AI Foundry & Private STT Infrastructure Coordinates ───────
    AI_FOUNDRY_API_VERSION    = os.getenv("AI_FOUNDRY_API_VERSION", "2024-06-01")
    AI_FOUNDRY_DEPLOYMENT     = os.getenv("AI_FOUNDRY_DEPLOYMENT", "gpt-4o-mini")
    USE_MANAGED_IDENTITY_SQL  = os.getenv("USE_MANAGED_IDENTITY_SQL", "false").lower() == "true"

    # ── Celery (non-credential parts) ─────────────────────────────────────
    CELERY_TASK_SERIALIZER   = os.getenv("CELERY_TASK_SERIALIZER", "json")
    CELERY_TASK_TIME_LIMIT   = int(os.getenv("CELERY_TASK_TIME_LIMIT", "600"))
    CELERY_TASK_ACKS_LATE    = os.getenv("CELERY_TASK_ACKS_LATE", "False").lower() == "true"
    CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2"))

    # Ensure upload folder exists at config load time (safe, not a secret)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # =========================================================================
    # SECRET ACCESSORS — always read from os.environ at call time.
    # bootstrap() in core.azure_identity populates os.environ from Key Vault
    # before these are ever called, so they always return the vault value in
    # production and the .env value in dev.
    # =========================================================================

    @classmethod
    def groq_api_key(cls) -> str | None:
        return os.getenv("GROQ_API_KEY")

    @classmethod
    def fireworks_api_key(cls) -> str | None:
        return os.getenv("FIREWORKS_API_KEY")

    @classmethod
    def mistral_api_key(cls) -> str | None:
        return os.getenv("MISTRAL_API_KEY")

    @classmethod
    def munsit_api_key(cls) -> str | None:
        return os.getenv("MUNSIT_API_KEY")

    @classmethod
    def openrouter_api_key(cls) -> str | None:
        return os.getenv("OPENROUTER_API_KEY")

    @classmethod
    def llm_api_key(cls) -> str | None:
        return os.getenv("LLM_API_KEY") or cls.openrouter_api_key()

    @classmethod
    def openrouter_api_keys(cls) -> list[str]:
        raw = os.getenv("OPENROUTER_API_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        primary = cls.llm_api_key()
        if primary and primary not in keys:
            keys.insert(0, primary)
        return keys

    # Per-service API key aliases (speech/refine/translation/extraction/questions)
    @classmethod
    def speech_api_key(cls) -> str | None:
        return os.getenv("speech")

    @classmethod
    def refine_api_key(cls) -> str | None:
        return os.getenv("refine")

    @classmethod
    def translate_api_key(cls) -> str | None:
        return os.getenv("translation")

    @classmethod
    def extraction_api_key(cls) -> str | None:
        return os.getenv("extraction")

    @classmethod
    def questions_api_key(cls) -> str | None:
        return os.getenv("questions")

    # Dashboard auth — replaced by Entra roles in Phase 1; kept for backward
    # compatibility with dev environments that still use basic auth.
    @classmethod
    def dashboard_username(cls) -> str | None:
        return os.getenv("DASHBOARD_USERNAME")   # default removed — explicit required

    @classmethod
    def dashboard_password(cls) -> str | None:
        return os.getenv("DASHBOARD_PASSWORD")   # default removed — explicit required

    # Database credentials
    @classmethod
    def patient_db_uid(cls) -> str | None:
        return os.getenv("PATIENT_DB_UID")

    @classmethod
    def patient_db_pwd(cls) -> str | None:
        return os.getenv("PATIENT_DB_PWD")

    # Celery credentials
    @classmethod
    def celery_broker_url(cls) -> str | None:
        return os.getenv("CELERY_BROKER_URL")

    @classmethod
    def celery_result_backend(cls) -> str | None:
        return os.getenv("CELERY_RESULT_BACKEND")

    @classmethod
    def celery_flower_password(cls) -> str | None:
        return os.getenv("CELERY_FLOWER_PASSWORD")

    # Phase 3: AI Foundry and Private STT
    @classmethod
    def ai_foundry_endpoint(cls) -> str | None:
        return os.getenv("AI_FOUNDRY_ENDPOINT")

    @classmethod
    def ai_foundry_key(cls) -> str | None:
        return os.getenv("AI_FOUNDRY_KEY") or os.getenv("AZURE_API_KEY")

    @classmethod
    def private_stt_endpoint(cls) -> str | None:
        return os.getenv("PRIVATE_STT_ENDPOINT")

    @classmethod
    def private_stt_key(cls) -> str | None:
        return os.getenv("PRIVATE_STT_KEY")

    # ── Per-model concurrency limit ───────────────────────────────────────
    @classmethod
    def llm_model_limit(cls, model: str) -> int:
        """Return the configured semaphore size for an OpenRouter model slug."""
        env_name = "LLM_LIMIT__" + re.sub(r"[^A-Za-z0-9]", "_", model).upper()
        raw_limit = os.getenv(env_name)
        limit = int(raw_limit) if raw_limit is not None else cls.LLM_LIMIT_DEFAULT
        if limit < 1:
            raise ValueError(f"{env_name} must be at least 1")
        return limit
