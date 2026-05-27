from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# Chấm điểm câu bằng tổng TF-IDF của các từ trong câu
class TfidfExtractor(BaseExtractor):
    name = "tfidf"

    def __init__(self, top_k_ratio: float = 0.2, ngram: tuple = (1, 2)):
        super().__init__(top_k_ratio = top_k_ratio)
        # Loại stopword tiếng Anh, dùng cả unigram + bigram
        self.vectorizer = TfidfVectorizer(stop_words = "english", ngram_range = ngram, max_df = 0.95)

    def score(self, sentences: List[Sentence]) -> List[float]:
        texts = [s.text for s in sentences]
        # Fit ngay trên doc hiện tại (mỗi câu là một "document")
        mat = self.vectorizer.fit_transform(texts)
        # Tổng điểm TF-IDF theo từng câu rồi chuẩn hoá theo độ dài để không thiên vị câu dài
        # np.asarray giúp Pyright nhận ra kết quả là ndarray để truy cập đúng
        row_sum = np.asarray(mat.sum(axis = 1)).ravel()  # type: ignore[attr-defined]
        lengths = [max(1, s.word_count) for s in sentences]
        return [float(row_sum[i] / lengths[i]) for i in range(len(sentences))]
