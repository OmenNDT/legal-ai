from typing import List, Optional
import numpy as np
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# Gộp điểm từ nhiều extractor (chuẩn hoá min-max rồi cộng có trọng số)
class EnsembleExtractor(BaseExtractor):
    name = "ensemble"

    def __init__(self, extractors: List[BaseExtractor], weights: Optional[List[float]] = None, top_k_ratio: float = 0.2):
        super().__init__(top_k_ratio = top_k_ratio)
        self.extractors = extractors
        if weights is None:
            weights = [1.0] * len(extractors)
        # Chuẩn hoá tổng trọng số == 1
        s = float(sum(weights))
        self.weights = [w / s for w in weights]

    @staticmethod
    def _minmax(arr):
        arr = np.asarray(arr, dtype = float)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def score(self, sentences: List[Sentence]) -> List[float]:
        total = np.zeros(len(sentences))
        for ex, w in zip(self.extractors, self.weights):
            s = self._minmax(ex.score(sentences))
            total += w * s
        return total.tolist()
