import os
import re
from dotenv import load_dotenv

load_dotenv()


def _configured_openrouter_api_keys() -> list[str]:
    """Return the primary OpenRouter key followed by any comma-separated fallbacks."""
    primary_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    configured_keys = os.getenv("OPENROUTER_API_KEYS", "").split(",")
    return list(dict.fromkeys(
        key.strip()
        for key in [primary_key, *configured_keys]
        if key and key.strip()
    ))


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
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Process-local settings for the shared asynchronous OpenRouter runtime.
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_API_KEYS = _configured_openrouter_api_keys()
    LLM_API_KEY = (
        os.getenv("LLM_API_KEY")
        or OPENROUTER_API_KEY
        or (OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else None)
    )
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
        """Return the configured semaphore size for an OpenRouter model slug."""
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
