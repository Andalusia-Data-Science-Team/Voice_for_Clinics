"""
Application configuration.

Secrets are accessed lazily through classmethods rather than being captured
at class-definition/import time. This allows core.azure_identity.bootstrap()
to populate os.environ from Azure Key Vault before secrets are accessed.

Non-secret infrastructure settings are loaded at import time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

_config_dir = Path(__file__).resolve().parent
_src_env = _config_dir.parent / ".env"
_root_env = _config_dir.parent.parent / ".env"

if _src_env.is_file():
    load_dotenv(_src_env)
elif _root_env.is_file():
    load_dotenv(_root_env)
else:
    load_dotenv()


class Config:
    """Base application configuration."""

    # -----------------------------------------------------------------------
    # General application settings
    # -----------------------------------------------------------------------

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg"}
    DEBUG = True
    TESTING = False
    PORT = 8586

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    DATABASE_PATH = "app_data.db"

    PATIENT_HISTORY_CSV_PATH = os.getenv(
        "PATIENT_HISTORY_CSV_PATH",
        "/home/ai/Workspace/Doha/MedVoice/src/data/Visit Level History.csv",
    )

    XLSX_PATH = os.getenv(
        "XLSX_PATH",
        "/home/ai/Workspace/Doha/MedVoice/src/data/"
        "Medical_Voice_Assistant_Expected_Fields_Expanded.xlsx",
    )

    # -----------------------------------------------------------------------
    # LLM runtime configuration (non-secret settings)
    # -----------------------------------------------------------------------

    LLM_BASE_URL = os.getenv(
        "LLM_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "45"))
    LLM_TOTAL_TIMEOUT_S = float(os.getenv("LLM_TOTAL_TIMEOUT_S", "300"))
    LLM_CONNECT_TIMEOUT_S = float(os.getenv("LLM_CONNECT_TIMEOUT_S", "5"))

    LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "4"))
    LLM_BACKOFF_MAX_S = float(os.getenv("LLM_BACKOFF_MAX_S", "8"))
    LLM_RETRY_AFTER_MAX_S = float(os.getenv("LLM_RETRY_AFTER_MAX_S", "30"))

    LLM_LIMIT_DEFAULT = int(os.getenv("LLM_LIMIT_DEFAULT", "8"))

    LLM_MAX_CONNECTIONS = int(
        os.getenv("LLM_MAX_CONNECTIONS", "100")
    )
    LLM_MAX_KEEPALIVE_CONNECTIONS = int(
        os.getenv("LLM_MAX_KEEPALIVE_CONNECTIONS", "20")
    )

    # -----------------------------------------------------------------------
    # Database configuration (non-credential settings)
    # -----------------------------------------------------------------------

    PATIENT_DB_DRIVER = os.getenv(
        "PATIENT_DB_DRIVER",
        "{ODBC Driver 17 for SQL Server}",
    )

    BUSINESS_UNITS = {
        "AHJ": {
            "server": "10.111.111.169",
            "database": "AHJ.Env.HCMS",
        },
        "AMH": {
            "server": "172.31.44.169",
            "database": "AMH.Env.HCMS",
        },
        "ANC": {
            "server": "172.31.44.169",
            "database": "ANC.Env.HCMS",
        },
    }

    # -----------------------------------------------------------------------
    # Azure AI Foundry and private STT infrastructure settings
    # -----------------------------------------------------------------------

    # LLM provider selection: "openrouter" (default) or "foundry".
    # Changing this variable is the only action needed to switch providers.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

    AI_FOUNDRY_API_VERSION = os.getenv(
        "AI_FOUNDRY_API_VERSION",
        "2025-01-01-preview",
    )

    # Foundry deployment names are configured per-model in core/model_registry.py
    # via FOUNDRY_DEPLOYMENT_LUNA etc. This top-level setting is kept for
    # backward-compat with APIM test_m1_verification.py but is no longer
    # used as a default model name in the inference path.
    AI_FOUNDRY_DEPLOYMENT = os.getenv(
        "AI_FOUNDRY_DEPLOYMENT",
        "",
    )

    USE_MANAGED_IDENTITY_SQL = (
        os.getenv("USE_MANAGED_IDENTITY_SQL", "false").lower() == "true"
    )

    # -----------------------------------------------------------------------
    # Celery runtime settings (non-secret settings)
    # -----------------------------------------------------------------------

    CELERY_TASK_SERIALIZER = os.getenv(
        "CELERY_TASK_SERIALIZER",
        "json",
    )

    CELERY_TASK_TIME_LIMIT = int(
        os.getenv("CELERY_TASK_TIME_LIMIT", "600")
    )

    CELERY_TASK_ACKS_LATE = (
        os.getenv("CELERY_TASK_ACKS_LATE", "False").lower() == "true"
    )

    CELERY_WORKER_CONCURRENCY = int(
        os.getenv("CELERY_WORKER_CONCURRENCY", "2")
    )

    # -----------------------------------------------------------------------
    # Upload directory
    # -----------------------------------------------------------------------

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # -----------------------------------------------------------------------
    # API keys
    # Read environment variables at call time to support Key Vault bootstrap.
    # -----------------------------------------------------------------------

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
        """Return the preferred LLM key, falling back to OpenRouter."""
        return os.getenv("LLM_API_KEY") or cls.openrouter_api_key()

    @classmethod
    def openrouter_api_keys(cls) -> list[str]:
        """
        Return the primary LLM key followed by configured fallback keys.

        Supports:
          LLM_API_KEY
          OPENROUTER_API_KEY
          OPENROUTER_API_KEYS=key1,key2,key3
        """
        raw_keys = os.getenv("OPENROUTER_API_KEYS", "")

        configured_keys = [
            key.strip()
            for key in raw_keys.split(",")
            if key.strip()
        ]

        primary = cls.llm_api_key()

        keys = []
        if primary:
            keys.append(primary)

        keys.extend(configured_keys)

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(keys))

    # -----------------------------------------------------------------------
    # Per-service API key aliases
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Dashboard authentication
    # No hardcoded credentials or default passwords.
    # -----------------------------------------------------------------------

    @classmethod
    def dashboard_username(cls) -> str | None:
        return os.getenv("DASHBOARD_USERNAME")

    @classmethod
    def dashboard_password(cls) -> str | None:
        return os.getenv("DASHBOARD_PASSWORD")

    # -----------------------------------------------------------------------
    # Database credentials
    # -----------------------------------------------------------------------

    @classmethod
    def patient_db_uid(cls) -> str | None:
        return os.getenv("PATIENT_DB_UID")

    @classmethod
    def patient_db_pwd(cls) -> str | None:
        return os.getenv("PATIENT_DB_PWD")

    # -----------------------------------------------------------------------
    # Celery credentials
    # -----------------------------------------------------------------------

    @classmethod
    def celery_broker_url(cls) -> str | None:
        return os.getenv("CELERY_BROKER_URL")

    @classmethod
    def celery_result_backend(cls) -> str | None:
        return os.getenv("CELERY_RESULT_BACKEND")

    @classmethod
    def celery_flower_password(cls) -> str | None:
        return os.getenv("CELERY_FLOWER_PASSWORD")

    # -----------------------------------------------------------------------
    # Azure AI Foundry and private STT credentials
    # -----------------------------------------------------------------------

    @classmethod
    def ai_foundry_endpoint(cls) -> str | None:
        return os.getenv("AI_FOUNDRY_ENDPOINT")

    @classmethod
    def ai_foundry_key(cls) -> str | None:
        return os.getenv("AI_FOUNDRY_KEY") or os.getenv("AZURE_API_KEY")

    @classmethod
    def ai_foundry_api_version(cls) -> str:
        return os.getenv("AI_FOUNDRY_API_VERSION", cls.AI_FOUNDRY_API_VERSION)

    @classmethod
    def private_stt_endpoint(cls) -> str | None:
        return os.getenv("PRIVATE_STT_ENDPOINT")

    @classmethod
    def private_stt_key(cls) -> str | None:
        return os.getenv("PRIVATE_STT_KEY")

    # -----------------------------------------------------------------------
    # Per-model LLM concurrency limits
    # -----------------------------------------------------------------------

    @classmethod
    def llm_model_limit(cls, model: str) -> int:
        """
        Return the semaphore limit for an OpenRouter model slug.

        Example:
            Model: openai/gpt-oss-20b
            Environment variable: LLM_LIMIT__OPENAI_GPT_OSS_20B
        """
        env_name = (
            "LLM_LIMIT__"
            + re.sub(r"[^A-Za-z0-9]", "_", model).upper()
        )

        raw_limit = os.getenv(env_name)
        limit = (
            int(raw_limit)
            if raw_limit is not None
            else cls.LLM_LIMIT_DEFAULT
        )

        if limit < 1:
            raise ValueError(f"{env_name} must be at least 1")

        return limit