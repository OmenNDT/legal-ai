"""Phần 2: Retrieval (Truy hồi tài liệu).

Luồng xử lý:
  ProcessedQuestion → Embedding (hoặc BM25) → Search → Top-k docs

Hỗ trợ 2 chế độ:
  - BM25 (mặc định, dùng search engine sẵn có)
  - Vector search (nếu có ChromaDB / sentence-transformers)
"""

import time
from typing import Optional

from src.rag_pipeline.contracts import ProcessedQuestion, RetrievedDocument, RetrievalResult
from src.search.search_engine import LegalSearchEngine


class LegalRetriever:
    """Truy hồi tài liệu pháp luật liên quan đến câu hỏi.

    Args:
        search_engine: LegalSearchEngine đã load sẵn index
        use_vector: Nếu True, thử dùng vector search (nếu có embedding)
        vector_weight: Trọng số cho vector score trong hybrid search (0-1)
    """

    def __init__(
        self,
        search_engine: Optional[LegalSearchEngine] = None,
        use_vector: bool = False,
        vector_weight: float = 0.3,
    ):
        self.search_engine = search_engine
        self.use_vector = False  # Disabled by default to avoid model download
        self.vector_weight = vector_weight
        self._embedding_model = None
        self._docs_cache: dict[str, str] = {}
        self._load_docs_cache()

    def _load_docs_cache(self):
        """Load document contents from legal_docs.json for retrieval."""
        try:
            from src.common.config import PROCESSED_DIR
            docs_path = PROCESSED_DIR / "legal_docs.json"
            if docs_path.exists():
                import json
                with open(docs_path, encoding="utf-8") as f:
                    docs = json.load(f)
                for i, doc in enumerate(docs):
                    content = doc.get("content", doc.get("segmented_content", doc.get("text", "")))
                    self._docs_cache[str(i)] = content
        except Exception:
            pass

    def _load_embedding_model(self):
        """Lazy-load sentence embedding model cho vector search."""
        if self._embedding_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except ImportError:
            self.use_vector = False  # Fallback to BM25

    def retrieve(
        self,
        question: ProcessedQuestion,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Truy hồi top-k tài liệu liên quan.

        Args:
            question: Câu hỏi đã qua tiền xử lý
            top_k: Số tài liệu cần lấy
            filters: Metadata filters (doc_type, date_range, ...)

        Returns:
            RetrievalResult với danh sách RetrievedDocument
        """
        start_time = time.time()

        # Ưu tiên segmented_text nếu có, nếu không dùng raw_text
        query_text = question.segmented_text or question.raw_text

        # Nếu có entities, bổ sung vào query để tăng độ chính xác
        enriched_query = self._enrich_query(query_text, question.entities)

        # Thực hiện search
        if self.search_engine is not None:
            results = self._search_bm25(enriched_query, top_k, filters)
        else:
            # Fallback: không có search engine
            results = []

        # Nếu bật vector search, kết hợp hybrid ranking
        if self.use_vector and results:
            results = self._hybrid_rerank(enriched_query, results)

        # Lọc theo metadata filters nếu có
        if filters:
            results = self._apply_filters(results, filters)

        latency = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=question.raw_text,
            documents=results[:top_k],
            total_found=len(results),
            retrieval_method="bm25" if not self.use_vector else "hybrid",
            latency_ms=latency,
        )

    def _enrich_query(self, query: str, entities: list) -> str:
        """Bổ sung entity names vào query để tăng recall."""
        extra_terms = []
        for ent in entities:
            if ent.label in ("LUAT", "THONG_TU", "NGHI_DINH", "DIEU", "KHOAN"):
                extra_terms.append(ent.text)
        if extra_terms:
            return f"{query} {' '.join(extra_terms)}"
        return query

    def _search_bm25(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict],
    ) -> list[RetrievedDocument]:
        """Search bằng BM25 từ LegalSearchEngine."""
        raw_results = self.search_engine.search(query, top_k=top_k * 2)  # Lấy nhiều hơn để rerank

        # Lazy load full docs if cache is empty
        if not self._docs_cache:
            self._load_docs_cache()

        docs = []
        for rank, r in enumerate(raw_results, start=1):
            doc_id = str(r.get("doc_id", rank))
            # Get content from cache first, then from result metadata
            content = self._docs_cache.get(doc_id, r.get("content", ""))
            
            doc = RetrievedDocument(
                doc_id=doc_id,
                content=content,
                metadata={
                    "title": r.get("title", ""),
                    "doc_type": r.get("type", ""),
                    "law_number": r.get("law_number", ""),
                    "issuer": r.get("issuer", ""),
                },
                score=float(r.get("score", 0.0)),
                rank=rank,
            )
            docs.append(doc)
        return docs

    def _hybrid_rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Kết hợp BM25 score + vector similarity."""
        self._load_embedding_model()
        if self._embedding_model is None:
            return docs

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            query_emb = self._embedding_model.encode([query])
            doc_texts = [d.content for d in docs]
            doc_embs = self._embedding_model.encode(doc_texts)

            similarities = cosine_similarity(query_emb, doc_embs)[0]

            # Normalize scores
            bm25_scores = np.array([d.score for d in docs])
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()

            vec_scores = np.array(similarities)
            if vec_scores.max() > 0:
                vec_scores = vec_scores / vec_scores.max()

            # Hybrid score
            hybrid = (1 - self.vector_weight) * bm25_scores + self.vector_weight * vec_scores

            for i, doc in enumerate(docs):
                doc.score = float(hybrid[i])

            docs.sort(key=lambda d: d.score, reverse=True)
            for i, doc in enumerate(docs, start=1):
                doc.rank = i

        except Exception:
            pass  # Giữ nguyên BM25 nếu lỗi

        return docs

    def _apply_filters(
        self,
        docs: list[RetrievedDocument],
        filters: dict,
    ) -> list[RetrievedDocument]:
        """Lọc documents theo metadata."""
        filtered = []
        for doc in docs:
            meta = doc.metadata or {}
            match = True
            for key, value in filters.items():
                if key in meta and meta[key] != value:
                    # Also check Vietnamese aliases
                    if key == "doc_type" and value == "luat" and meta[key] == "law":
                        continue
                    if key == "doc_type" and value == "nghi_dinh" and meta[key] == "decree":
                        continue
                    if key == "doc_type" and value == "thong_tu" and meta[key] == "circular":
                        continue
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered
