import os
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
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    _openrouter_extra_raw = os.getenv("OPENROUTER_API_KEYS", "")
    OPENROUTER_API_KEYS = [k.strip() for k in _openrouter_extra_raw.split(",") if k.strip()]
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY not in OPENROUTER_API_KEYS:
        OPENROUTER_API_KEYS.insert(0, OPENROUTER_API_KEY)

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
