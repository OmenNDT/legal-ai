import os
import torch
from pathlib import Path

# Cấu hình toàn cục cho pipeline
class Settings:
    # Đường dẫn gốc của project
    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = ROOT_DIR / "data"
    TXT_DIR: Path = DATA_DIR / "full_contract_txt"
    CSV_FILE: Path = DATA_DIR / "master_clauses.csv"
    LABEL_DIR: Path = DATA_DIR / "label_group_xlsx"

    # Thư mục xuất kết quả
    OUT_DIR: Path = ROOT_DIR / "outputs"
    OUT_EXTRACTIVE: Path = OUT_DIR / "extractive"
    OUT_ABSTRACTIVE: Path = OUT_DIR / "abstractive"
    OUT_HYBRID: Path = OUT_DIR / "hybrid"
    OUT_EVAL: Path = OUT_DIR / "eval"
    LOG_DIR: Path = ROOT_DIR / "logs"
    CACHE_DIR: Path = ROOT_DIR / "backend" / "cache"
    MODEL_CACHE: Path = CACHE_DIR / "models"
    SUMMARY_CACHE: Path = CACHE_DIR / "summaries"

    # Tham số extractive - giữ 50% câu để cover nhiều clause nhưng vẫn vừa số chunk
    TOP_K_RATIO: float = 0.85
    MIN_SENT_LEN: int = 5
    MAX_SENT_LEN: int = 80

    # Tham số abstractive (BART) - cân bằng giữa độ dài và tốc độ
    BART_MODEL: str = "facebook/bart-large-cnn"
    BART_MAX_INPUT: int = 1024
    BART_MAX_OUTPUT: int = 512
    BART_MIN_OUTPUT: int = 350
    # Beams=2 thay vì 4 để giảm latency một nửa, recall không thay đổi nhiều
    BART_NUM_BEAMS: int = 5
    # Ghép thẳng chunk_summaries làm output (recall cao hơn hẳn so với hierarchical)
    BART_KEEP_CHUNKS: bool = True

    # Mô hình embedding dùng cho KMeans extractor
    SBERT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Độ chồng lấn khi chia chunk văn bản dài (3-5 câu là đủ)
    CHUNK_OVERLAP: int = 6

    # Label-guided reranker: dùng clause trong master_clauses.csv để bổ sung câu cho extractive
    # Chỉ áp dụng khi request có doc_id (eval/benchmark hoặc UI chọn doc có sẵn)
    LABEL_GUIDED: bool = True
    LABEL_SIM_THRESHOLD: float = 0.35
    LABEL_MAX_EXTRA_RATIO: float = 0.4

    # Cấu hình Flask API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 9020

    # Tự động phát hiện thiết bị GPU/CPU
    @staticmethod
    def device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    # Tạo trước các thư mục đầu ra nếu chưa tồn tại
    @classmethod
    def ensure_dirs(cls):
        for d in [cls.OUT_EXTRACTIVE, cls.OUT_ABSTRACTIVE, cls.OUT_HYBRID, cls.OUT_EVAL, cls.LOG_DIR, cls.MODEL_CACHE, cls.SUMMARY_CACHE]:
            d.mkdir(parents = True, exist_ok = True)

# Cho phép override qua biến môi trường
def get_settings():
    s = Settings()
    s.API_PORT = int(os.getenv("API_PORT", s.API_PORT))
    s.BART_MODEL = os.getenv("BART_MODEL", s.BART_MODEL)
    s.ensure_dirs()
    return s