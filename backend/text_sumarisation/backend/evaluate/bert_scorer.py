from typing import List, Tuple
from torch import Tensor
from bert_score import score as bscore

# BERTScore - đánh giá theo nghĩa thay vì chỉ trùng từ
class BertScoreEvaluator:
    def __init__(self, lang: str = "en", device: str = "cpu", model_type: str = "roberta-large"):
        self.lang = lang
        self.device = device
        self.model_type = model_type

    def score(self, predictions: List[str], references: List[str]) -> Tuple[float, float, float]:
        # bscore() trả về 3 tensor (P, R, F) khi return_hash = False (mặc định)
        result = bscore(
            predictions,
            references,
            lang = self.lang,
            device = self.device,
            model_type = self.model_type,
            verbose = False
        )
        p: Tensor
        r: Tensor
        f: Tensor
        p, r, f = result  # type: ignore[misc]
        return float(p.mean().item()), float(r.mean().item()), float(f.mean().item())
