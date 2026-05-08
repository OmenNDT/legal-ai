"""Phần 2: Retrieval (Truy hồi tài liệu).

Luồng xử lý:
  ProcessedQuestion → Embedding (hoặc BM25) → Search → Top-k docs

Hỗ trợ 2 chế độ:
  - BM25 (mặc định, dùng search engine sẵn có)
  - Vector search (nếu có ChromaDB / sentence-transformers)
"""

import logging
import time
from typing import Optional

from src.rag_pipeline.contracts import ProcessedQuestion, RetrievedDocument, RetrievalResult
from src.rag_pipeline.query_expansion import expand_query
from src.rag_pipeline.retrieval_utils import (
    convert_hybrid_results,
    convert_vector_results,
    deduplicate_docs,
    HybridReranker,
)
from src.search.hybrid_search import HybridSearch
from src.search.search_engine import LegalSearchEngine
from src.search.vector_store import VectorStore

logger = logging.getLogger(__name__)


class LegalRetriever:
    """Truy hồi tài liệu pháp luật liên quan đến câu hỏi.

    Args:
        search_engine: LegalSearchEngine đã load sẵn index
        use_vector: Nếu True, thử dùng vector search (nếu có embedding)
        vector_weight: Trọng số cho vector score trong hybrid search (0-1)
        vector_store: VectorStore instance for ChromaDB search
        hybrid_search: HybridSearch instance for RRF fusion
    """

    def __init__(
        self,
        search_engine: Optional[LegalSearchEngine] = None,
        use_vector: bool = False,
        vector_weight: float = 0.3,
        vector_store: Optional[VectorStore] = None,
        hybrid_search: Optional[HybridSearch] = None,
    ):
        self.search_engine = search_engine
        self.use_vector = use_vector
        self.vector_weight = vector_weight
        self.vector_store = vector_store
        self.hybrid_search = hybrid_search
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

    def retrieve(
        self,
        question: ProcessedQuestion,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Truy hồi top-k tài liệu liên quan."""
        start_time = time.time()
        query_text = question.segmented_text or question.raw_text
        expanded_queries = expand_query(query_text, question.entities)

        results: list[RetrievedDocument] = []

        # Prefer HybridSearch when available and vector enabled
        if self.use_vector and self.hybrid_search is not None:
            try:
                raw = self.hybrid_search.search(expanded_queries, n_results=top_k * 2)
                results = convert_hybrid_results(raw)
            except Exception as exc:
                logger.warning("HybridSearch failed: %s", exc)

        # Fallback 1: BM25 (+ optional local vector rerank)
        if not results and self.search_engine is not None:
            for eq in expanded_queries:
                results.extend(self._search_bm25(eq, top_k, filters))
            results = deduplicate_docs(results)

            if self.use_vector and self.vector_store and self.vector_store.is_available:
                try:
                    vec_results = self.vector_store.search(query_text, n_results=top_k * 2)
                    vec_docs = convert_vector_results(vec_results)
                    merged = {d.doc_id: d for d in results}
                    for vd in vec_docs:
                        if vd.doc_id in merged:
                            if vd.score > merged[vd.doc_id].score:
                                merged[vd.doc_id] = vd
                        else:
                            merged[vd.doc_id] = vd
                    results = list(merged.values())
                    reranker = HybridReranker(vector_weight=self.vector_weight)
                    results = reranker.rerank(query_text, results)
                except Exception as exc:
                    logger.warning("Vector search failed: %s", exc)

        # Fallback 2: pure vector search when no BM25 engine
        if not results and self.search_engine is None and self.use_vector and self.vector_store and self.vector_store.is_available:
            try:
                vec_results = self.vector_store.search(query_text, n_results=top_k * 2)
                results = convert_vector_results(vec_results)
            except Exception as exc:
                logger.warning("Vector search failed: %s", exc)

        if not results and self.search_engine is None and not (self.vector_store and self.vector_store.is_available):
            logger.warning("No search engine available for retrieval")

        # Apply filters after all retrieval paths
        if filters:
            results = self._apply_filters(results, filters)

        if filters:
            results = self._apply_filters(results, filters)

        results.sort(key=lambda d: d.score, reverse=True)
        for i, doc in enumerate(results, start=1):
            doc.rank = i

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
        raw_results = self.search_engine.search(query, top_k=top_k * 2)
        if not self._docs_cache:
            self._load_docs_cache()
        docs = []
        for rank, r in enumerate(raw_results, start=1):
            doc_id = str(r.get("doc_id", rank))
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
