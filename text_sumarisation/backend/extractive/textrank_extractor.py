from typing import List
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# TextRank: xây đồ thị câu, chạy PageRank để tìm câu trung tâm
class TextRankExtractor(BaseExtractor):
    name = "textrank"

    def __init__(self, top_k_ratio: float = 0.2, damping: float = 0.85):
        super().__init__(top_k_ratio = top_k_ratio)
        self.damping = damping

    def score(self, sentences: List[Sentence]) -> List[float]:
        texts = [s.text for s in sentences]
        # Vector hoá câu rồi tính cosine làm trọng số cạnh
        vec = TfidfVectorizer(stop_words = "english")
        mat = vec.fit_transform(texts)
        sim = cosine_similarity(mat)
        # Đặt đường chéo về 0 để câu không tự kết với chính nó
        np.fill_diagonal(sim, 0)
        # Tạo đồ thị có trọng số
        g = nx.from_numpy_array(sim)
        # Chạy PageRank, trả về điểm cho từng câu
        pr = nx.pagerank(g, alpha = self.damping, max_iter = 200)
        return [float(pr.get(i, 0.0)) for i in range(len(sentences))]
