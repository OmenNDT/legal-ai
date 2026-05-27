from dataclasses import dataclass, asdict
from typing import List
from rouge_score import rouge_scorer

# Bộ ba ROUGE
@dataclass
class RougeScore:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    rouge1_p: float = 0.0
    rouge1_r: float = 0.0

    def to_dict(self):
        return asdict(self)

# Tính ROUGE giữa predicted summary và reference
class RougeEvaluator:
    def __init__(self, use_stemmer: bool = True):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer = use_stemmer)

    def score(self, prediction: str, reference: str) -> RougeScore:
        if not prediction or not reference:
            return RougeScore(0.0, 0.0, 0.0, 0.0, 0.0)
        r = self.scorer.score(reference, prediction)
        return RougeScore(
            rouge1_f = float(r["rouge1"].fmeasure),
            rouge2_f = float(r["rouge2"].fmeasure),
            rougeL_f = float(r["rougeL"].fmeasure),
            rouge1_p = float(r["rouge1"].precision),
            rouge1_r = float(r["rouge1"].recall)
        )

    # Tính trên cả batch và trả về điểm trung bình
    def score_batch(self, predictions: List[str], references: List[str]):
        assert len(predictions) == len(references)
        all_scores = [self.score(p, r) for p, r in zip(predictions, references)]
        n = max(1, len(all_scores))
        avg = RougeScore(
            rouge1_f = sum(s.rouge1_f for s in all_scores) / n,
            rouge2_f = sum(s.rouge2_f for s in all_scores) / n,
            rougeL_f = sum(s.rougeL_f for s in all_scores) / n,
            rouge1_p = sum(s.rouge1_p for s in all_scores) / n,
            rouge1_r = sum(s.rouge1_r for s in all_scores) / n
        )
        return avg, all_scores
