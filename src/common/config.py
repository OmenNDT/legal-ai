import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

QA_DATA_PATH = RAW_DIR / "yuiTC_sample.json"
LEGAL_DATA_PATH = RAW_DIR / "uts_vlc_processed.json"

INTENT_MODEL_DIR = MODELS_DIR / "intent_classifier"
NER_MODEL_DIR = MODELS_DIR / "ner_tagger"
SCORER_MODEL_DIR = MODELS_DIR / "sentence_scorer"
LORA_CHECKPOINT_PATH = MODELS_DIR / "lora_ke_toan" / "best_model.pt"
SEARCH_INDEX_PATH = PROCESSED_DIR / "search_index.json"
KG_PATH = PROCESSED_DIR / "knowledge_graph.gpickle"

PHOBERT_MODEL = "vinai/phobert-base"
MAX_SEQ_LENGTH = 256

INTENT_LABELS = [
    "hoi_dieu_khoan",
    "hoi_luat_moi",
    "tra_cuu_thong_tu",
    "tra_cuu_nghi_dinh",
    "tra_cuu_luat",
    "hoi_hieu_luc",
    "hoi_sua_doi",
    "hoi_dinh_nghia",
    "hoi_thu_tuc",
    "hoi_hinh_phat",
    "hoi_quyen_loi",
    "hoi_nghia_vu",
    "hoi_bieu_phi",
    "hoi_thoi_han",
    "hoi_doi_tuong",
    "hoi_dieu_kien",
    "hoi_phan_biet",
    "hoi_so_sanh",
    "hoi_tom_tat",
    "hoi_tong_hop",
]

ENTITY_TYPES = [
    "LUAT",
    "THONG_TU",
    "NGHI_DINH",
    "DIEU",
    "KHOAN",
    "DIEM",
    "CO_QUAN",
    "KHAISUAT",
    "NGAY_THANG",
]

NER_LABELS = ["O"] + [f"{prefix}-{etype}" for etype in ENTITY_TYPES for prefix in ["B", "I"]]

SUMMARY_WEIGHTS = {
    "relevance": 0.4,
    "centrality": 0.4,
    "position": 0.2,
}

TEXTRANK_DAMPING = 0.85
TEXTRANK_MAX_ITER = 100
TEXTRANK_TOLERANCE = 1e-5

BM25_K1 = 1.5
BM25_B = 0.75
SEARCH_TOP_K = 10
FUZZY_MAX_DIST = 2

RELATION_TYPES = [
    "DUA_TREN",
    "THAM_CHIEU",
    "HET_HIEU_LUC",
    "THAY_THE",
    "SUA_DOI_BO_SUNG",
    "HUONG_DAN",
    "CHUA",
]

DOCUMENT_TYPES = [
    "hien_phap",
    "bo_luat",
    "luat",
    "nghi_dinh",
    "thong_tu",
    "quyet_dinh",
    "nghi_quyet",
]

TRAIN_BATCH_SIZE = 32
SCORER_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
SCORER_LEARNING_RATE = 2e-5
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
FP16 = True

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "legal_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
