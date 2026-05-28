from typing import List, Optional
import numpy as np
from backend.extractive.base import BaseExtractor
from backend.preprocess.splitter import Sentence

# Ensemble là một phương pháp trong học máy và trích xuất thông tin, trong đó nhiều mô hình hoặc thuật toán được kết hợp lại để cải thiện hiệu suất tổng thể. Thay vì dựa vào một mô hình duy nhất, ensemble sử dụng nhiều mô hình khác nhau để đưa ra dự đoán hoặc đánh giá, và sau đó kết hợp các kết quả này theo một cách nào đó. Trong ngữ cảnh của trích xuất thông tin, ensemble có thể giúp tận dụng điểm mạnh của nhiều phương pháp khác nhau để xác định những câu quan trọng nhất trong văn bản, từ đó tạo ra một bản tóm tắt chính xác và hiệu quả hơn.

# Gộp điểm từ nhiều extractor (chuẩn hoá min-max rồi cộng có trọng số)
# Chuẩn hoá min-max giúp đưa điểm về cùng thang 0-1 để cộng có ý nghĩa hơn, tránh trường hợp extractor này cho điểm rất cao còn extractor khác cho điểm rất thấp làm mất tác dụng của extractor có điểm thấp. Việc cộng có trọng số (weights) cho phép điều chỉnh tầm quan trọng của từng extractor trong việc tính điểm cuối cùng, giúp tối ưu hóa hiệu suất của ensemble dựa trên đặc điểm của từng phương pháp trích xuất thông tin.
# Công thức chuẩn hoá min-max: x' = (x - min) / (max - min)
# Công thức tính điểm final_score:
# final_score[i] = Σ (w_j * (score_j[i] - min_j) / (max_j - min_j))

class EnsembleExtractor(BaseExtractor):
    name = "ensemble"

    def __init__(self, extractors: List[BaseExtractor], weights: Optional[List[float]] = None, top_k_ratio: float = 0.2):
        super().__init__(top_k_ratio = top_k_ratio)
        self.extractors = extractors # Danh sách các extractor thành phần
        if weights is None:
            weights = [1.0] * len(extractors) # Nếu không có trọng số, mặc định là 1 cho tất cả
        # Chuẩn hoá tổng trọng số == 1
        s = float(sum(weights))
        self.weights = [w / s for w in weights] # Chuẩn hoá trọng số để tổng == 1

    @staticmethod
    def _minmax(arr):
        arr = np.asarray(arr, dtype = float)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo) # Chuẩn hoá min-max về thang 0-1

    def score(self, sentences: List[Sentence]) -> List[float]:
        total = np.zeros(len(sentences))
        for ex, w in zip(self.extractors, self.weights):
            s = self._minmax(ex.score(sentences)) # Chuẩn hoá điểm của extractor này về thang 0-1
            total += w * s # Cộng điểm có trọng số của từng extractor vào tổng điểm cuối cùng
        return total.tolist()
    