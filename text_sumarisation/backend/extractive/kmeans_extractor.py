from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# Gom cụm câu bằng embedding sentence-transformers rồi chọn câu gần tâm cụm nhất
class KMeansExtractor(BaseExtractor):
    name = "kmeans"

    def __init__(self, top_k_ratio: float = 0.2, sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        super().__init__(top_k_ratio = top_k_ratio)
        self.sbert_name = sbert_model
        self.device = device
        self._model = None

    # Lười khởi tạo model để không tốn RAM khi không dùng
    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.sbert_name, device = self.device)
        return self._model

    def score(self, sentences: List[Sentence]) -> List[float]:
        model = self._ensure_model()
        texts = [s.text for s in sentences]
        emb = model.encode(texts, batch_size = 32, show_progress_bar = False, convert_to_numpy = True)
        n = len(sentences)
        # Số cụm == số câu cần giữ lại
        k = max(self.min_keep, int(round(n * self.top_k_ratio)))
        k = min(k, n)
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        labels = km.fit_predict(emb)
        centers = km.cluster_centers_
        # Tính khoảng cách của mỗi câu tới tâm cụm của nó
        scores = np.zeros(n)
        for i in range(n):
            d = np.linalg.norm(emb[i] - centers[labels[i]])
            # Càng gần tâm điểm càng cao
            scores[i] = 1.0 / (1.0 + d)
        return scores.tolist()
