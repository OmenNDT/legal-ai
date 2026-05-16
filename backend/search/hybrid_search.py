class HybridSearch:
    RRF_K = 60

    def __init__(self, bm25, vector_store, bm25_weight: float = 0.5, vector_weight: float = 0.5):
        self._bm25 = bm25
        self._vector_store = vector_store
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight

    def search(self, queries: list, n_results: int = 50) -> list:
        bm25_map: dict = {}
        vector_map: dict = {}

        for query in queries:
            for rank, result in enumerate(self._bm25.search(query, top_k=n_results)):
                cid = result.get("chunk_id", result.get("doc_id", 0))
                if cid not in bm25_map or rank < bm25_map[cid]["rank"]:
                    bm25_map[cid] = {"rank": rank, "data": result, "score": result.get("score", result.get("bm25_score", 0.0))}

            # Only search vector store if it's available
            if hasattr(self._vector_store, 'is_available') and self._vector_store.is_available:
                try:
                    for rank, result in enumerate(self._vector_store.search(query, n_results=n_results)):
                        cid = result.get("chunk_id", 0)
                        if cid not in vector_map or rank < vector_map[cid]["rank"]:
                            vector_map[cid] = {"rank": rank, "data": result, "score": result.get("vector_score", 0.0)}
                except Exception:
                    pass

        all_ids = set(bm25_map) | set(vector_map)
        scored = []

        for cid in all_ids:
            bm25_rrf = 1.0 / (self.RRF_K + bm25_map[cid]["rank"] + 1) if cid in bm25_map else 0.0
            vec_rrf = 1.0 / (self.RRF_K + vector_map[cid]["rank"] + 1) if cid in vector_map else 0.0
            hybrid_score = self._bm25_weight * bm25_rrf + self._vector_weight * vec_rrf

            source = (bm25_map.get(cid) or vector_map.get(cid))["data"].copy()
            source["chunk_id"] = cid
            source["bm25_score"] = bm25_map[cid]["score"] if cid in bm25_map else 0.0
            source["vector_score"] = vector_map[cid]["score"] if cid in vector_map else 0.0
            source["hybrid_score"] = round(hybrid_score, 6)
            scored.append(source)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:n_results]
