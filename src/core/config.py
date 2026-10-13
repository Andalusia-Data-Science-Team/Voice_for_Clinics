from __future__ import annotations

import os
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}
    DEBUG = True
    TESTING = False
    PORT = 8586
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MUNSIT_API_KEY = os.getenv("MUNSIT_API_KEY")

    # ── Provider switches (M2) ────────────────────────────────────────────────
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "foundry").lower()
    STT_PROVIDER = os.getenv("STT_PROVIDER", "munsit").lower()

    # ── Azure AI Foundry settings ─────────────────────────────────────────────
    AI_FOUNDRY_ENDPOINT = (
        os.getenv("AI_FOUNDRY_ENDPOINT")
        or os.getenv("FOUNDRY_BASE_URL")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or "https://andalusiafoundry.openai.azure.com/openai/v1"
    )
    AI_FOUNDRY_KEY = (
        os.getenv("AI_FOUNDRY_KEY")
        or os.getenv("FOUNDRY_API_KEY")
        or os.getenv("AZURE_API_KEY")
    )
    AI_FOUNDRY_API_VERSION = os.getenv("AI_FOUNDRY_API_VERSION", "2024-06-01")
    AI_FOUNDRY_AUTH_METHOD = os.getenv("AI_FOUNDRY_AUTH_METHOD", "api_key").lower()

    @classmethod
    def ai_foundry_auth_method(cls) -> str:
        return os.getenv("AI_FOUNDRY_AUTH_METHOD", cls.AI_FOUNDRY_AUTH_METHOD).lower()

    @classmethod
    def ai_foundry_endpoint(cls) -> str:
        return (
            os.getenv("AI_FOUNDRY_ENDPOINT")
            or os.getenv("FOUNDRY_BASE_URL")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or cls.AI_FOUNDRY_ENDPOINT
        )

    @classmethod
    def ai_foundry_key(cls) -> str | None:
        return (
            os.getenv("AI_FOUNDRY_KEY")
            or os.getenv("FOUNDRY_API_KEY")
            or os.getenv("AZURE_API_KEY")
            or cls.AI_FOUNDRY_KEY
        )

    @classmethod
    def ai_foundry_api_version(cls) -> str:
        return os.getenv("AI_FOUNDRY_API_VERSION", cls.AI_FOUNDRY_API_VERSION)

    FOUNDRY_DEPLOYMENT_DEEPSEEK = os.getenv("FOUNDRY_DEPLOYMENT_DEEPSEEK", "DeepSeek-V3.2")

    @classmethod
    def foundry_deployment_deepseek(cls) -> str:
        return os.getenv("FOUNDRY_DEPLOYMENT_DEEPSEEK", cls.FOUNDRY_DEPLOYMENT_DEEPSEEK)

    # Process-local settings for the shared asynchronous Foundry runtime.
    LLM_BASE_URL = os.getenv("LLM_BASE_URL") or AI_FOUNDRY_ENDPOINT
    LLM_API_KEY = os.getenv("LLM_API_KEY") or AI_FOUNDRY_KEY
    LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "45"))
    LLM_TOTAL_TIMEOUT_S = float(os.getenv("LLM_TOTAL_TIMEOUT_S", "300"))
    LLM_CONNECT_TIMEOUT_S = float(os.getenv("LLM_CONNECT_TIMEOUT_S", "5"))
    LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "4"))
    LLM_BACKOFF_MAX_S = float(os.getenv("LLM_BACKOFF_MAX_S", "8"))
    LLM_RETRY_AFTER_MAX_S = float(os.getenv("LLM_RETRY_AFTER_MAX_S", "30"))
    LLM_LIMIT_DEFAULT = int(os.getenv("LLM_LIMIT_DEFAULT", "8"))
    LLM_MAX_CONNECTIONS = int(os.getenv("LLM_MAX_CONNECTIONS", "100"))
    LLM_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("LLM_MAX_KEEPALIVE_CONNECTIONS", "20"))

    @classmethod
    def llm_model_limit(cls, model: str) -> int:
        """Return the configured semaphore size for a model deployment."""
        env_name = "LLM_LIMIT__" + re.sub(r"[^A-Za-z0-9]", "_", model).upper()
        raw_limit = os.getenv(env_name)
        limit = int(raw_limit) if raw_limit is not None else cls.LLM_LIMIT_DEFAULT
        if limit < 1:
            raise ValueError(f"{env_name} must be at least 1")
        return limit

    SPEECH_API_KEY = os.getenv("speech")
    REFINE_API_KEY = os.getenv("refine")
    TRANSLATE_API_KEY = os.getenv("translation")
    EXTRACTION_API_KEY = os.getenv("extraction")
    QUESTIONS_API_KEY = os.getenv("questions")

    DATABASE_PATH = "app_data.db"
    PATIENT_HISTORY_CSV_PATH = os.getenv("PATIENT_HISTORY_CSV_PATH", "/home/ai/Workspace/Doha/MedVoice/src/data/Visit Level History.csv")
    XLSX_PATH = os.getenv("XLSX_PATH", "/home/ai/Workspace/Doha/MedVoice/src/data/Medical_Voice_Assistant_Expected_Fields_Expanded.xlsx")
    DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "DohaMahmoud")
    DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "1811")
    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    BUSINESS_UNITS = {
        "AHJ": {"server": "10.111.111.169", "database": "AHJ.Env.HCMS"},
        "AMH": {"server": "172.31.44.169",  "database": "AMH.Env.HCMS"},
        "ANC": {"server": "172.31.44.169",  "database": "ANC.Env.HCMS"},
    }

    PATIENT_DB_UID = os.getenv("PATIENT_DB_UID")
    PATIENT_DB_PWD = os.getenv("PATIENT_DB_PWD")
    PATIENT_DB_DRIVER = os.getenv("PATIENT_DB_DRIVER", "{ODBC Driver 17 for SQL Server}")
