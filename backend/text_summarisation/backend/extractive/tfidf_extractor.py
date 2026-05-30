from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# TF-IDF (Term Frequency - Inverse Document Frequency) là một phương pháp phổ biến trong xử lý ngôn ngữ tự nhiên để đánh giá mức độ quan trọng của một từ trong một tài liệu (document) so với một tập hợp tài liệu (corpus). Nó được tính bằng cách nhân tần suất xuất hiện của từ trong tài liệu (TF) với logarit của nghịch đảo tần suất xuất hiện của từ trong tập hợp tài liệu (IDF). TF-IDF giúp xác định những từ nào là quan trọng và có ý nghĩa trong việc tóm tắt nội dung của văn bản.
# Công thức: TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
# Trong đó:
# - t là từ cần đánh giá.
# - d là tài liệu chứa từ t.
# - D là tập hợp tài liệu (corpus) mà chúng ta đang xét.
# - TF(t, d) là tần suất xuất hiện của từ t trong tài liệu d.
# - IDF(t, D) = log(N / (1 + DF(t))) là logarit của nghịch đảo tần suất xuất hiện của từ t trong tập hợp tài liệu D, với DF(t) là số tài liệu chứa từ t và N là tổng số tài liệu trong tập hợp D.
# Công thức IDF có thể được điều chỉnh để tránh chia cho 0 bằng cách thêm 1 vào mẫu số, trở thành IDF(t, D) = log(N / (1 + DF(t))). Điều này đảm bảo rằng nếu một từ xuất hiện trong tất cả các tài liệu (DF(t) = N), thì IDF sẽ là log(N / (1 + N)) = log(N / (N + 1)) = log(1 / (1 + 1/N)) ≈ 0, thay vì log(0) không xác định. Điều này giúp đảm bảo rằng các từ phổ biến không bị đánh giá quá cao và vẫn có thể được sử dụng trong việc tóm tắt nội dung của văn bản.
# - N là tổng số tài liệu trong tập hợp D.
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
        lengths = [max(1, s.word_count) for s in sentences] # Chuẩn hoá theo độ dài câu (số từ) để không thiên vị câu dài
        return [float(row_sum[i] / lengths[i]) for i in range(len(sentences))]
