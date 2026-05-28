"""Phần 3: Augment (Bổ sung ngữ cảnh).

Luồng xử lý:
  RetrievalResult → Rerank → Build Context Window → AugmentedContext

Mục tiêu: Tối ưu ngữ cảnh trước khi đưa vào LLM/Generator.
"""

import time
from typing import Optional

from backend.rag_pipeline.contracts import (
    RetrievalResult,
    RetrievedDocument,
    AugmentedContext,
)
from backend.common.config import CROSS_ENCODER_MODEL


class ContextAugmenter:
    """Bổ sung và tối ưu ngữ cảnh cho generation.

    Args:
        max_tokens: Số token tối đa trong context window
        overlap_tokens: Số token overlap giữa các đoạn
        use_reranker: Có dùng cross-encoder reranker không
        reranker_model: Tên model cross-encoder
    """

    def __init__(
        self,
        max_tokens: int = 1024,
        overlap_tokens: int = 50,
        use_reranker: bool = True,
        reranker_model: str = CROSS_ENCODER_MODEL,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self._reranker = None

    def _load_reranker(self):
        """Lazy-load cross-encoder reranker."""
        if self._reranker is not None or not self.use_reranker:
            return
        try:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._reranker = CrossEncoder(self.reranker_model, device=device)
        except Exception:
            self.use_reranker = False

    def augment(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        top_k: int = 5,
    ) -> AugmentedContext:
        """Tạo augmented context từ retrieval results.

        Args:
            question: Câu hỏi gốc
            retrieval_result: Kết quả từ Phần 2
            top_k: Số document tối đa đưa vào context

        Returns:
            AugmentedContext đã được rerank và ghép
        """
        start_time = time.time()
        docs = retrieval_result.documents[:top_k]

        # Bước 1: Rerank nếu có reranker
        if self.use_reranker and len(docs) > 1:
            docs = self._rerank(question, docs)

        # Bước 2: Build context text từ các đoạn đã chọn
        context_text, selected_docs = self._build_context(question, docs)

        # Bước 3: Tính token count (ước lượng)
        token_count = len(context_text.split())

        latency = (time.time() - start_time) * 1000

        return AugmentedContext(
            original_question=question,
            context_text=context_text,
            documents=selected_docs,
            rerank_scores=[d.score for d in selected_docs],
            token_count=token_count,
            context_strategy="concat",
        )

    def _rerank(
        self,
        question: str,
        docs: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Rerank documents bằng cross-encoder."""
        self._load_reranker()
        if self._reranker is None:
            return docs

        try:
            pairs = [[question, d.content] for d in docs]
            scores = self._reranker.predict(pairs)

            for doc, score in zip(docs, scores):
                doc.score = float(score)

            docs.sort(key=lambda d: d.score, reverse=True)
            for i, doc in enumerate(docs, start=1):
                doc.rank = i
        except Exception:
            pass

        return docs

    def _build_context(
        self,
        question: str,
        docs: list[RetrievedDocument],
    ) -> tuple[str, list[RetrievedDocument]]:
        """Ghép các đoạn văn bản thành context window.

        Chiến lược:
        1. Lấy đoạn đầu tiên (relevant nhất) đầy đủ
        2. Các đoạn tiếp theo: truncate nếu vượt quá max_tokens
        3. Thêm metadata header cho mỗi đoạn
        """
        if not docs:
            return "Không tìm thấy tài liệu liên quan.", []

        parts = []
        selected = []
        total_tokens = 0
        max_tokens_per_doc = self.max_tokens // min(len(docs), 3)

        for i, doc in enumerate(docs):
            # Tạo header với metadata
            doc_type = doc.metadata.get("doc_type", "Tài liệu")
            doc_name = doc.metadata.get("name", f"Tài liệu {i+1}")
            header = f"[{doc_type}] {doc_name}"

            # Truncate nội dung nếu cần
            content = doc.content.strip()
            content_tokens = len(content.split())
            if content_tokens > max_tokens_per_doc and i > 0:
                words = content.split()
                content = " ".join(words[:max_tokens_per_doc]) + "..."
                content_tokens = max_tokens_per_doc

            part = f"{header}\n{content}\n"
            part_tokens = len(part.split())

            if total_tokens + part_tokens > self.max_tokens and i > 0:
                break

            parts.append(part)
            selected.append(doc)
            total_tokens += part_tokens

        context = f"Câu hỏi: {question}\n\n" + "\n---\n".join(parts)
        return context, selected

    def augment_with_kg(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        reasoner,  # LegalReasoner
        top_k: int = 5,
    ) -> AugmentedContext:
        """Augment với Knowledge Graph context (nếu có KG)."""
        base = self.augment(question, retrieval_result, top_k)

        if reasoner is None or not retrieval_result.documents:
            return base

        # Thêm KG reasoning cho document đầu tiên
        top_doc = retrieval_result.documents[0]
        doc_id = top_doc.doc_id

        try:
            validity = reasoner.check_validity(doc_id)
            if validity.reasoning_steps:
                kg_context = f"\n[Thông tin bổ sung từ Knowledge Graph]\n"
                kg_context += f"- {validity.answer}\n"
                for step in validity.reasoning_steps[:3]:
                    kg_context += f"- {step}\n"

                base.context_text += kg_context
                base.context_strategy = "concat+kg"
        except Exception:
            pass

        return base
