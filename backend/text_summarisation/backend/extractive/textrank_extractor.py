from typing import List
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# TextRank là một thuật toán trích xuất thông tin dựa trên nguyên lý của PageRank, được sử dụng để đánh giá mức độ quan trọng của các câu trong một văn bản. Thuật toán này xây dựng một đồ thị mà các nút đại diện cho các câu và các cạnh đại diện cho sự tương đồng giữa các câu. Điểm số của mỗi câu được tính dựa trên điểm số của các câu liên kết với nó, với trọng số của các cạnh được xác định bởi mức độ tương đồng giữa các câu. TextRank giúp xác định những câu nào là trung tâm và quan trọng nhất trong văn bản, từ đó có thể được sử dụng để tạo ra một bản tóm tắt ngắn gọn và hiệu quả.
# PageRank là một thuật toán được sử dụng để xếp hạng các trang dựa trên số lượng và chất lượng của các liên kết đến trang đó. Trong ngữ cảnh của TextRank, PageRank được áp dụng để đánh giá mức độ quan trọng của các câu trong một văn bản dựa trên sự tương đồng giữa chúng. Các câu có nhiều liên kết đến từ các câu khác sẽ được đánh giá cao hơn, cho thấy rằng chúng có vai trò trung tâm trong việc truyền đạt thông tin quan trọng trong văn bản. Điều này giúp TextRank xác định những câu nào nên được giữ lại trong bản tóm tắt cuối cùng.
# Công thức PageRank: PR(A) = (1 - d) + d * (PR(B1)/L(B1) + ... + PR(Bn)/L(Bn))
# Trong đó:
# - PR(A) là điểm PageRank của câu A.
# - d là hệ số damping (thường là 0.85), đại diện cho xác suất người dùng tiếp tục theo liên kết thay vì dừng lại.
# - B1, ..., Bn là các câu liên kết đến câu A.
# - PR(Bi) là điểm PageRank của câu Bi.
# - L(Bi) là số lượng liên kết ra khỏi câu Bi.
# Cosine similarity là một phép đo được sử dụng để đánh giá mức độ tương đồng giữa hai vector trong không gian đa chiều. Trong ngữ cảnh của TextRank, cosine similarity được sử dụng để tính toán mức độ tương đồng giữa các câu dựa trên biểu diễn vector của chúng (thường là TF-IDF). Công thức tính cosine similarity giữa hai vector A và B là: cosine_similarity(A, B) = (A . B) / (||A|| * ||B||), trong đó A . B là tích vô hướng của hai vector và ||A||, ||B|| là độ dài của các vector. Kết quả của cosine similarity nằm trong khoảng từ -1 đến 1, với giá trị gần 1 cho thấy hai câu rất giống nhau, giá trị gần 0 cho thấy chúng không tương đồng, và giá trị gần -1 cho thấy chúng hoàn toàn khác nhau. Trong TextRank, các cạnh giữa các câu được tạo ra dựa trên mức độ tương đồng này, giúp xác định những câu nào có liên quan chặt chẽ với nhau trong văn bản.

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
        scores = [float(pr.get(i, 0.0)) for i in range(len(sentences))]
        # Xuất graph để FE vẽ: chỉ giữ các cạnh đủ mạnh để hình không rối
        self._last_extra["graph"] = self._build_graph_payload(sentences, sim, scores)
        return scores

    # Đóng gói graph nhẹ để gửi sang FE (không gửi numpy)
    def _build_graph_payload(self, sentences, sim, scores):
        n = len(sentences)
        # Ngưỡng cạnh: giữ top edge để hình vẽ rõ. Lấy 75th percentile, tối thiểu 0.05.
        flat = sim[np.triu_indices(n, k = 1)] if n > 1 else np.array([])
        if flat.size > 0:
            thr = float(max(np.percentile(flat, 75), 0.05))
        else:
            thr = 0.05
        edges = []
        # Giới hạn số cạnh để FE không lag (max ~ 4*n hoặc 400)
        max_edges = max(4 * n, 50)
        # Sắp các cặp theo similarity giảm dần
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                w = float(sim[i, j])
                if w >= thr:
                    pairs.append((w, i, j))
        pairs.sort(reverse = True)
        for w, i, j in pairs[:max_edges]:
            edges.append({"source": i, "target": j, "weight": round(w, 4)})
        nodes = [
            {
                "id": i,
                "idx": s.idx,
                "score": round(scores[i], 6),
                "preview": (s.text[:120] + "...") if len(s.text) > 120 else s.text,
                "words": s.word_count,
                "picked": False
            }
            for i, s in enumerate(sentences)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "edge_threshold": round(thr, 4),
            "damping": self.damping
        }
