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
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    SPEECH_API_KEY = os.getenv("speech")
    REFINE_API_KEY = os.getenv("refine")
    TRANSLATE_API_KEY = os.getenv("translation")
    EXTRACTION_API_KEY = os.getenv("extraction")
    QUESTIONS_API_KEY = os.getenv("questions")

    DATABASE_PATH = "app_data.db"
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
