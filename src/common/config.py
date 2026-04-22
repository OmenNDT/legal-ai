"""Configuration: paths, model names, hyperparameters."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────
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
SEARCH_INDEX_PATH = PROCESSED_DIR / "search_index.json"
KG_PATH = PROCESSED_DIR / "knowledge_graph.gpickle"

# ── Model ──────────────────────────────────────────────
PHOBERT_MODEL = "vinai/phobert-base"
MAX_SEQ_LENGTH = 256

# ── Intent Classification ─────────────────────────────
INTENT_LABELS = [
    "hoi_dieu_khoan",        # hỏi nội dung điều khoản
    "hoi_luat_moi",          # hỏi luật mới ban hành
    "tra_cuu_thong_tu",      # tra cứu thông tư
    "tra_cuu_nghi_dinh",     # tra cứu nghị định
    "tra_cuu_luat",          # tra cứu luật
    "hoi_hieu_luc",          # hỏi còn hiệu lực không
    "hoi_sua_doi",           # hỏi văn bản sửa đổi
    "hoi_dinh_nghia",        # hỏi định nghĩa/khái niệm
    "hoi_thu_tuc",           # hỏi thủ tục
    "hoi_hinh_phat",         # hỏi hình phạt/xử phạt
    "hoi_quyen_loi",         # hỏi quyền lợi
    "hoi_nghia_vu",          # hỏi nghĩa vụ
    "hoi_bieu_phi",           # hỏi biểu phí/lệ phí
    "hoi_thoi_han",          # hỏi thời hạn/thời hiệu
    "hoi_doi_tuong",         # hỏi đối tượng áp dụng
    "hoi_dieu_kien",         # hỏi điều kiện
    "hoi_phan_biet",         # hỏi phân biệt/sự khác biệt
    "hoi_so_sanh",           # hỏi so sánh
    "hoi_tom_tat",           # yêu cầu tóm tắt
    "hoi_tong_hop",          # hỏi tổng hợp
]

# ── NER Entity Types (BIO scheme) ─────────────────────
ENTITY_TYPES = [
    "LUAT",        # Luật (Law)
    "THONG_TU",    # Thông tư (Circular)
    "NGHI_DINH",   # Nghị định (Decree)
    "DIEU",        # Điều (Article)
    "KHOAN",       # Khoản (Clause)
    "DIEM",        # Điểm (Point)
    "CO_QUAN",     # Cơ quan (Agency/Organization)
    "KHAISUAT",    # Khái niệm pháp lý (Legal concept)
    "NGAY_THANG",  # Ngày tháng (Date)
]

# B-I-O tags
NER_LABELS = ["O"] + [f"{prefix}-{etype}" for etype in ENTITY_TYPES for prefix in ["B", "I"]]

# ── Summarizer ──────────────────────────────────────────
SUMMARY_WEIGHTS = {
    "relevance": 0.4,   # cosine(query, sentence)
    "centrality": 0.4,  # TextRank PageRank
    "position": 0.2,    # sentence position bonus
}

TEXTRANK_DAMPING = 0.85
TEXTRANK_MAX_ITER = 100
TEXTRANK_TOLERANCE = 1e-5

# ── Search ──────────────────────────────────────────────
BM25_K1 = 1.5
BM25_B = 0.75
SEARCH_TOP_K = 10
FUZZY_MAX_DIST = 2

# ── Knowledge Graph ─────────────────────────────────────
RELATION_TYPES = [
    "DUA_TREN",          # dựa trên (is pursuant to)
    "THAM_CHIEU",        # tham chiếu (refers to)
    "HET_HIEU_LUC",      # hết hiệu lực (expires)
    "THAY_THE",          # thay thế (supersedes)
    "SUA_DOI_BO_SUNG",   # sửa đổi/bổ sung (amends/supplements)
    "HUONG_DAN",         # hướng dẫn (guides)
    "CHUA",              # chứa (contains)
]

DOCUMENT_TYPES = [
    "hien_phap",    # Hiến pháp (Constitution)
    "bo_luat",      # Bộ luật (Code)
    "luat",         # Luật (Law)
    "nghi_dinh",    # Nghị định (Decree)
    "thong_tu",     # Thông tư (Circular)
    "quyet_dinh",   # Quyết định (Decision)
    "nghi_quyet",   # Nghị quyết (Resolution)
]

# ── Training ────────────────────────────────────────────
TRAIN_BATCH_SIZE = 32
SCORER_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
SCORER_LEARNING_RATE = 2e-5
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
FP16 = True