"""Environment configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API
    API_TITLE = "Affiliate Product Analysis API"
    API_VERSION = "0.1.0"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./affiliate.db")
    
    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # Image Processing
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
    
    # Affiliate APIs
    SHOPEE_API_KEY = os.getenv("SHOPEE_API_KEY")
    LAZADA_API_KEY = os.getenv("LAZADA_API_KEY")
    TIKTOK_SHOP_API_KEY = os.getenv("TIKTOK_SHOP_API_KEY")
    
    # Cache
    CACHE_TTL = 3600  # 1 hour
    
    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

settings = Settings()
