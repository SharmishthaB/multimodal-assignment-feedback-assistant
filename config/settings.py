"""Configuration settings for MAFA.

Loads environment variables and provides app-wide settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Settings:
    """Application settings."""

    # Directories
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    CHROMADB_DIR = PROJECT_ROOT / "data" / "chromadb"

    # Model paths
    VLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", str(PROJECT_ROOT / "models" / "phi_3"))

    # ChromaDB
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", str(CHROMADB_DIR))

    # Embeddings
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    # Device
    DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

    # Quantization
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = PROJECT_ROOT / "logs" / "mafa.log"

    @classmethod
    def ensure_dirs(cls):
        """Create required directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.CHROMADB_DIR, cls.LOG_FILE.parent]:
            directory.mkdir(parents=True, exist_ok=True)


# Initialize
settings = Settings()
settings.ensure_dirs()
