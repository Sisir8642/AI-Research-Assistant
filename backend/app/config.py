"""
Central configuration module.
Reads all settings from environment variables via .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os

class Settings(BaseSettings):
    # --- API Keys ---
    groq_api_key: str = os.getenv("GROQ_API_KEY")

    # --- LLM ---
    groq_model: str = "llama3-8b-8192"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector DB ---
    vector_db_type: str = "faiss"  # "faiss" or "chroma"

    # --- Paths ---
    docs_dir: str = "data/docs"
    vector_db_dir: str = "data/vector_db"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k_results: int = 5

    # --- CORS ---
    allowed_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    def ensure_dirs(self):
        """Create required directories if they don't exist."""
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_db_dir).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Use this everywhere: from app.config import get_settings
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings