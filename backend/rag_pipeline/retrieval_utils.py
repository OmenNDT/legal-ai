"""Utility functions for LegalRetriever."""

from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.rag_pipeline.contracts import RetrievedDocument
from backend.search.vector_store import VectorStore


def convert_hybrid_results(raw: list[dict]) -> list[RetrievedDocument]:
    """Convert HybridSearch RRF results to RetrievedDocument."""
    docs = []
    for rank, r in enumerate(raw, start=1):
        doc_id = str(r.get("chunk_id", r.get("doc_id", rank)))
        docs.append(
            RetrievedDocument(
                doc_id=doc_id,
                content=r.get("chunk_text", r.get("content", "")),
                metadata={
                    "title": r.get("document_title", ""),
                    "doc_type": r.get("document_type", ""),
                    "law_number": r.get("law_number", ""),
                    "issuer": r.get("issuer", ""),
                },
                score=float(r.get("hybrid_score", 0.0)),
                rank=rank,
            )
        )
    return docs


def convert_vector_results(raw: list[dict]) -> list[RetrievedDocument]:
    """Convert VectorStore search results to RetrievedDocument."""
    docs = []
    for rank, r in enumerate(raw, start=1):
        doc_id = str(r.get("chunk_id", rank))
        docs.append(
            RetrievedDocument(
                doc_id=doc_id,
                content=r.get("chunk_text", r.get("content", "")),
                metadata={
                    "title": r.get("document_title", ""),
                    "doc_type": r.get("document_type", ""),
                },
                score=float(r.get("vector_score", 0.0)),
                rank=rank,
            )
        )
    return docs


def deduplicate_docs(docs: list[RetrievedDocument]) -> list[RetrievedDocument]:
    """Deduplicate by doc_id keeping highest score."""
    seen: dict[str, RetrievedDocument] = {}
    for d in docs:
        if d.doc_id in seen:
            if d.score > seen[d.doc_id].score:
                seen[d.doc_id] = d
        else:
            seen[d.doc_id] = d
    return list(seen.values())


class HybridReranker:
    """Local fallback hybrid rerank using sentence-transformers."""

    def __init__(self, vector_weight: float = 0.3):
        self.vector_weight = vector_weight
        self._embedding_model = None

    def _load_model(self):
        if self._embedding_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except ImportError:
            pass

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Combine BM25 score + vector similarity."""
        self._load_model()
        if self._embedding_model is None:
            return docs

        try:
            query_emb = self._embedding_model.encode([query])
            doc_texts = [d.content for d in docs]
            doc_embs = self._embedding_model.encode(doc_texts)
            similarities = cosine_similarity(query_emb, doc_embs)[0]

            bm25_scores = np.array([d.score for d in docs])
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()

            vec_scores = np.array(similarities)
            if vec_scores.max() > 0:
                vec_scores = vec_scores / vec_scores.max()

            hybrid = (1 - self.vector_weight) * bm25_scores + self.vector_weight * vec_scores
            for i, doc in enumerate(docs):
                doc.score = float(hybrid[i])

            docs.sort(key=lambda d: d.score, reverse=True)
            for i, doc in enumerate(docs, start=1):
                doc.rank = i
        except Exception:
            pass
        return docs
