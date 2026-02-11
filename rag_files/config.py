from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf_files"
TEXT_DIR = DATA_DIR / "text_files"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
SCORE_THRESHOLD = 0.3
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
