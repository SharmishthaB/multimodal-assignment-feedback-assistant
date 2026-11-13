"""
ChromaDB Builder Script
Run once in environment setup to compile rubric markdown and lecture slides into ChromaDB.
"""
import logging
from src.config import CHROMA_PATH, RUBRIC_PATH
from src.repository import build_vector_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PDF_DIR = "/kaggle/input/datasets/sharmishthab/ml-slides"

if __name__ == "__main__":
    print("⏳ Building ChromaDB collections...")
    build_vector_database(
        chroma_path=CHROMA_PATH,
        rubric_path=RUBRIC_PATH,
        pdf_dir=PDF_DIR,
    )
    print("🎉 ChromaDB build complete!")
