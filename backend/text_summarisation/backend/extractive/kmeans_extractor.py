from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# Kmeans extractor là một phương pháp trích xuất thông tin dựa trên thuật toán K-Means clustering, được sử dụng để nhóm các câu trong một văn bản thành các cụm dựa trên sự tương đồng của chúng. Thuật toán này hoạt động bằng cách phân chia các câu thành K cụm sao cho các câu trong cùng một cụm có sự tương đồng cao hơn với nhau so với các câu ở các cụm khác. Điểm số của mỗi câu được tính dựa trên khoảng cách của nó đến tâm cụm của cụm mà nó thuộc về, với những câu gần tâm cụm hơn được đánh giá cao hơn. KMeans extractor giúp xác định những câu nào là trung tâm và quan trọng nhất trong văn bản, từ đó có thể được sử dụng để tạo ra một bản tóm tắt ngắn gọn và hiệu quả.

# Gom cụm câu bằng embedding sentence-transformers rồi chọn câu gần tâm cụm nhất
class KMeansExtractor(BaseExtractor):
    name = "kmeans"

    def __init__(self, top_k_ratio: float = 0.2, sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        super().__init__(top_k_ratio = top_k_ratio)
        self.sbert_name = sbert_model
        self.device = device
        self._model = None

    # Cơ chế lazy initialization để không tốn RAM khi không dùng
    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.sbert_name, device = self.device)
        return self._model

    def score(self, sentences: List[Sentence]) -> List[float]:
        model = self._ensure_model()
        texts = [s.text for s in sentences]

        # Tạo embedding cho từng câu, sau đó gom cụm bằng KMeans
        # Embedding là vector số học thể hiện ngữ nghĩa của câu, giúp KMeans nhóm các câu có ý nghĩa tương tự lại với nhau. Câu nào gần tâm cụm hơn sẽ được đánh giá là quan trọng hơn vì nó đại diện tốt hơn cho nội dung chung của cụm đó.
        emb = model.encode(texts, batch_size = 32, show_progress_bar = False, convert_to_numpy = True)
        n = len(sentences)
        # Gom cụm KMeans
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
            # Càng gần tâm, điểm càng cao
            scores[i] = 1.0 / (1.0 + d)
        return scores.tolist()
