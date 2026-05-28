"""RAG Extract Configuration (integrated into legal-ai)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "rag_pipeline.db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# RAG storage paths
UPLOAD_DIR = Path(os.getenv("RAG_DOCUMENTS_DIR", str(DATA_DIR / "uploads" / "raw")))
PROCESSED_DIR = Path(os.getenv("RAG_PROCESSED_DIR", str(DATA_DIR / "uploads" / "processed")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma")))
LIGHTRAG_DIR = Path(os.getenv("LIGHTRAG_DIR", str(DATA_DIR / "lightrag")))

# Ensure directories exist
for d in [UPLOAD_DIR, PROCESSED_DIR, CHROMA_DIR, LIGHTRAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
