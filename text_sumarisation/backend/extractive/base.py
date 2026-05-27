from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from backend.preprocess.splitter import Sentence

# Kết quả trích xuất: danh sách câu giữ thứ tự gốc + điểm tương ứng
@dataclass
class ExtractResult:
    method: str
    sentences: List[Sentence]
    scores: List[float]

    def as_text(self) -> str:
        return " ".join(s.text for s in self.sentences)

# Interface chung cho mọi extractor
class BaseExtractor(ABC):
    name: str = "base"

    def __init__(self, top_k_ratio: float = 0.2, min_keep: int = 5):
        # Tỉ lệ câu giữ lại (mặc định 20%) và số câu tối thiểu
        self.top_k_ratio = top_k_ratio
        self.min_keep = min_keep

    # Mỗi extractor cụ thể implement hàm này
    @abstractmethod
    def score(self, sentences: List[Sentence]) -> List[float]:
        ...

    # Chọn top-K theo điểm rồi sắp lại theo thứ tự gốc của câu
    def extract(self, sentences: List[Sentence]) -> ExtractResult:
        if not sentences:
            return ExtractResult(method = self.name, sentences = [], scores = [])
        scores = self.score(sentences)
        k = max(self.min_keep, int(round(len(sentences) * self.top_k_ratio)))
        k = min(k, len(sentences))
        # Lấy index của K câu có điểm cao nhất
        ranked = sorted(range(len(sentences)), key = lambda i: scores[i], reverse = True)[:k]
        ranked.sort()
        picked = [sentences[i] for i in ranked]
        picked_scores = [float(scores[i]) for i in ranked]
        return ExtractResult(method = self.name, sentences = picked, scores = picked_scores)
