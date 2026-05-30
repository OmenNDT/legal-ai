from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from backend.preprocess.splitter import Sentence

# Kết quả trích xuất: danh sách câu giữ thứ tự gốc + điểm tương ứng
@dataclass
class ExtractResult:
    method: str
    sentences: List[Sentence]
    scores: List[float]
    # Dữ liệu phụ trợ tuỳ extractor (vd: graph của TextRank để vẽ trên UI)
    extra: Optional[Dict[str, Any]] = field(default_factory = dict)

    def as_text(self) -> str:
        return " ".join(s.text for s in self.sentences)

# Interface chung cho mọi extractor
class BaseExtractor(ABC):
    name: str = "base"

    def __init__(self, top_k_ratio: float = 0.2, min_keep: int = 5):
        # Tỉ lệ câu giữ lại (mặc định 20%) và số câu tối thiểu
        self.top_k_ratio = top_k_ratio
        self.min_keep = min_keep
        # Dữ liệu phụ trợ extractor có thể set trong score() để gắn vào ExtractResult
        self._last_extra: Dict[str, Any] = {}

    # Mỗi extractor cụ thể implement hàm này
    @abstractmethod
    def score(self, sentences: List[Sentence]) -> List[float]:
        ...

    # Chọn top-K theo điểm rồi sắp lại theo thứ tự gốc của câu
    def extract(self, sentences: List[Sentence]) -> ExtractResult:
        if not sentences:
            return ExtractResult(method = self.name, sentences = [], scores = [], extra = {})
        self._last_extra = {}
        scores = self.score(sentences)
        # Tính K (Số câu cần giữ lại)
        k = max(self.min_keep, int(round(len(sentences) * self.top_k_ratio)))
        k = min(k, len(sentences))
        # Lấy index của K câu có điểm cao nhất (Sắp xếp điểm giảm dần rồi lấy top K)
        ranked = sorted(range(len(sentences)), key = lambda i: scores[i], reverse = True)[:k]
        # Đánh dấu các câu được chọn để đồ thị biết tô màu (TextRank dùng)
        picked_set = set(ranked)
        if "graph" in self._last_extra:
            for node in self._last_extra["graph"].get("nodes", []):
                node["picked"] = node["id"] in picked_set
        # Sắp lại index theo thứ tự gốc của câu để giữ trật tự khi nối thành văn bản
        ranked.sort()
        picked = [sentences[i] for i in ranked]
        picked_scores = [float(scores[i]) for i in ranked]
        return ExtractResult(method = self.name, sentences = picked, scores = picked_scores, extra = self._last_extra)
