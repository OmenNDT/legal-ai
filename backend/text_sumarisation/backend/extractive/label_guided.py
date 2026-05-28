from typing import List, Optional
from sentence_transformers import SentenceTransformer
from backend.extractive.base import ExtractResult
from backend.evaluate.reference_builder import ReferenceBuilder
from backend.preprocess.splitter import Sentence

# Reranker dùng nhãn CUAD làm tín hiệu yếu để bổ sung câu cho extractive result.
# Với mỗi clause của doc, tìm câu gần nhất trong raw text bằng SBERT;
# Nếu câu đó chưa được extractor chọn, thêm vào (tới max_extra) để cover clause.
class LabelGuidedReranker:
    def __init__(
        self,
        reference_builder: ReferenceBuilder,
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        sim_threshold: float = 0.35,
        max_extra_ratio: float = 0.4
    ):
        self.ref_builder = reference_builder
        self.sbert_name = sbert_model
        self.device = device
        self.sim_threshold = sim_threshold
        self.max_extra_ratio = max_extra_ratio
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.sbert_name, device = self.device)
        return self._model

    # Lấy list clause riêng lẻ (chưa join) của 1 doc
    def _clauses_for(self, doc_id: str) -> List[str]:
        df = self.ref_builder._ensure_loaded()
        row = df[df["doc_id"] == doc_id]
        if row.empty:
            return []
        row0 = row.iloc[0]
        out: List[str] = []
        seen = set()
        for col in self.ref_builder._clause_columns():
            for p in self.ref_builder._parse_cell(row0.get(col)):
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    # Bổ sung câu liên quan tới clause vào ExtractResult (giữ thứ tự gốc)
    def rerank(self, doc_id: Optional[str], all_sentences: List[Sentence], ext: ExtractResult) -> ExtractResult:
        if not doc_id or not all_sentences:
            return ext
        clauses = self._clauses_for(doc_id)
        if not clauses:
            return ext

        model = self._ensure_model()
        sent_texts = [s.text for s in all_sentences]
        sent_emb = model.encode(sent_texts, batch_size = 64, show_progress_bar = False, convert_to_numpy = True, normalize_embeddings = True)
        clause_emb = model.encode(clauses, batch_size = 32, show_progress_bar = False, convert_to_numpy = True, normalize_embeddings = True)

        # sim[i, j] = similarity câu i với clause j
        sim = sent_emb @ clause_emb.T
        # Với mỗi clause, lấy câu có sim cao nhất
        best_sent_per_clause = sim.argmax(axis = 0)
        best_sim_per_clause = sim.max(axis = 0)

        already_idx = {s.idx for s in ext.sentences}
        # Số câu tối đa được thêm vào (tránh phình BART input)
        max_extra = max(1, int(round(len(all_sentences) * self.max_extra_ratio)))

        # Ứng viên: (sent_idx, sim) sort giảm dần theo sim
        candidates = []
        for c_idx, s_idx in enumerate(best_sent_per_clause):
            s_idx = int(s_idx)
            score = float(best_sim_per_clause[c_idx])
            if score < self.sim_threshold:
                continue
            if s_idx in already_idx:
                continue
            candidates.append((s_idx, score))
        # Khử trùng lặp (nhiều clause map về cùng 1 câu) - giữ điểm cao nhất
        best_by_sent: dict[int, float] = {}
        for s_idx, sc in candidates:
            if sc > best_by_sent.get(s_idx, -1.0):
                best_by_sent[s_idx] = sc
        ranked_extra = sorted(best_by_sent.items(), key = lambda x: x[1], reverse = True)[:max_extra]
        if not ranked_extra:
            return ext

        extra_idx = {s_idx for s_idx, _ in ranked_extra}
        extra_scores = dict(ranked_extra)

        # Gộp + sort lại theo idx gốc để giữ trình tự đọc
        merged_idx = sorted(already_idx | extra_idx)
        idx_to_sent = {s.idx: s for s in all_sentences}
        old_score_by_idx = {s.idx: sc for s, sc in zip(ext.sentences, ext.scores)}
        new_sentences = [idx_to_sent[i] for i in merged_idx if i in idx_to_sent]
        new_scores = [old_score_by_idx.get(i, extra_scores.get(i, 0.0)) for i in merged_idx if i in idx_to_sent]

        # Giữ extra (vd: graph TextRank) và cập nhật picked theo set mới
        new_extra = dict(ext.extra or {})
        if "graph" in new_extra:
            picked_set = {s.idx for s in new_sentences}
            new_extra["graph"] = {
                **new_extra["graph"],
                "nodes": [
                    {**node, "picked": node["idx"] in picked_set}
                    for node in new_extra["graph"].get("nodes", [])
                ],
            }
        return ExtractResult(
            method = f"{ext.method}+label",
            sentences = new_sentences,
            scores = new_scores,
            extra = new_extra
        )
