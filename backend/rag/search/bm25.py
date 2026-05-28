import math
from backend.search.inverted_index import InvertedIndex

class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self._index = index
        self.k1 = k1
        self.b = b
        self._idf: dict = {}
        self._compute_idf()

    def _compute_idf(self):
        N = self._index.doc_count
        for term in self._index._index:
            df = len(self._index.search(term))
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list:
        terms = self._index._tokenize(query)
        if not terms:
            return []
        candidates = set()
        for t in terms:
            for p in self._index.search(t):
                candidates.add(p.doc_id)
        scored = []
        for doc_id in candidates:
            score = self._score_doc(terms, doc_id)
            if score > 0:
                meta = self._index.get_document_info(doc_id)
                scored.append({"doc_id": doc_id, "chunk_id": doc_id, "score": round(score, 4), **meta})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _score_doc(self, terms: list, doc_id: int) -> float:
        score = 0.0
        doc_len = self._index._doc_lengths.get(doc_id, 0)
        avgdl = self._index.avg_doc_length
        for term in terms:
            idf = self._idf.get(term, 0)
            tf = 0
            for p in self._index.search(term):
                if p.doc_id == doc_id:
                    tf = p.term_frequency()
                    break
            if tf == 0:
                continue
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * doc_len / max(avgdl, 1))
            score += idf * num / den
        return score

    def explain(self, query: str, doc_id: int) -> dict:
        terms = self._index._tokenize(query)
        doc_len = self._index._doc_lengths.get(doc_id, 0)
        avgdl = self._index.avg_doc_length
        breakdown = []
        total = 0.0
        for term in terms:
            idf = self._idf.get(term, 0)
            tf = 0
            for p in self._index.search(term):
                if p.doc_id == doc_id:
                    tf = p.term_frequency()
                    break
            if tf == 0:
                breakdown.append({"term": term, "idf": round(idf, 4), "tf": 0, "score": 0})
                continue
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * doc_len / max(avgdl, 1))
            ts = idf * num / den
            total += ts
            breakdown.append({"term": term, "idf": round(idf, 4), "tf": tf, "score": round(ts, 4)})
        return {"doc_id": doc_id, "total": round(total, 4), "breakdown": breakdown}
