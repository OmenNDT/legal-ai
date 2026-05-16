"""Phần 4: Generation (Sinh câu trả lời).

Luồng xử lý:
  AugmentedContext → PhoBERT/LLM → GeneratedAnswer

Mục tiêu: Sinh câu trả lời chất lượng cao, có trích dẫn nguồn.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.rag_pipeline.contracts import AugmentedContext, GeneratedAnswer, Citation
from src.rag_pipeline.generation_utils import (
    split_into_sentences,
    score_sentences,
    build_answer_from_sentences,
)

try:
    from src.llm.client import LLMClient
except ImportError:
    LLMClient = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class LegalAnswerGenerator:
    """Sinh câu trả lời pháp luật từ ngữ cảnh đã augment.

    Hỗ trợ 2 chế độ:
      - "extractive": Trích xuất câu trả lời từ context (mặc định, không cần LLM)
      - "llm": Dùng LLM client (Ollama/OpenAI-compatible)
    """

    def __init__(
        self,
        generation_mode: str = "extractive",
        llm_client: Optional[LLMClient] = None,
    ):
        self.generation_mode = generation_mode
        self.llm_client = llm_client

    def generate(
        self,
        context: AugmentedContext,
        max_length: int = 512,
    ) -> GeneratedAnswer:
        """Sinh câu trả lời từ augmented context."""
        start_time = time.time()

        if self.generation_mode == "llm" and self.llm_client is not None:
            answer = self._generate_llm(context, max_length)
        else:
            answer = self._generate_extractive(context)

        citations = self._extract_citations(context)
        reasoning = self._build_reasoning(context)
        latency = (time.time() - start_time) * 1000

        return GeneratedAnswer(
            answer_text=answer,
            confidence=self._compute_confidence(context),
            citations=citations,
            reasoning_steps=reasoning,
            generation_method=self.generation_mode,
            latency_ms=latency,
        )

    def _generate_extractive(self, context: AugmentedContext) -> str:
        """Trích xuất câu trả lời từ context (không dùng generative model)."""
        question = context.original_question
        ctx_text = context.context_text

        if "Câu hỏi:" in ctx_text:
            parts = ctx_text.split("\n\n", 1)
            if len(parts) == 2:
                ctx_text = parts[1]

        sentences = split_into_sentences(ctx_text)
        if not sentences:
            return "Xin lỗi, không tìm thấy thông tin phù hợp để trả lời câu hỏi."

        scored = score_sentences(question, sentences)
        top_sentences = [s for s, _ in scored[:5]]
        return build_answer_from_sentences(question, top_sentences)

    def _generate_llm(self, context: AugmentedContext, max_length: int) -> str:
        """Generate answer via LLM client (Ollama-compatible API)."""
        prompt = self._build_prompt(context)
        system = (
            "Bạn là chuyên gia pháp lý Việt Nam. Hãy phân tích câu hỏi và trả lời dựa trên thông tin được cung cấp.\n"
            "Cấu trúc câu trả lời phải có các phần:\n"
            "**Căn cứ pháp lý:**\n"
            "**Phân tích:**\n"
            "**Kết luận:**\n"
            "**Khuyến nghị:**"
        )
        try:
            return self.llm_client.complete(
                prompt,
                system=system,
                max_tokens=max_length,
                temperature=0.1,
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return self._generate_extractive(context)

    def _build_prompt(self, context: AugmentedContext) -> str:
        """Xây dựng prompt cho generative model."""
        question = context.original_question
        ctx = context.context_text

        prompt = (
            f"Dựa trên các thông tin pháp luật sau, hãy trả lời câu hỏi:\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Thông tin tham khảo:\n{ctx}\n\n"
            f"Câu trả lời:"
        )
        return prompt

    def _extract_citations(self, context: AugmentedContext) -> list[Citation]:
        """Trích xuất citations từ documents."""
        citations = []
        for doc in context.documents[:3]:
            citation = Citation(
                doc_id=doc.doc_id,
                doc_name=doc.metadata.get("name", doc.metadata.get("title", "Unknown")),
                excerpt=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                relevance_score=doc.score,
            )
            citations.append(citation)
        return citations

    def _build_reasoning(self, context: AugmentedContext) -> list[str]:
        """Xây dựng các bước reasoning."""
        steps = [
            f"1. Nhận câu hỏi: '{context.original_question[:80]}...'" if len(context.original_question) > 80 else f"1. Nhận câu hỏi: '{context.original_question}'",
            f"2. Truy hồi {len(context.documents)} tài liệu liên quan",
        ]
        for i, doc in enumerate(context.documents[:3], 3):
            doc_name = doc.metadata.get("name", doc.metadata.get("title", f"Tài liệu {i-2}"))
            steps.append(f"{i}. Tham khảo: {doc_name} (score: {doc.score:.3f})")
        if self.generation_mode == "llm":
            steps.append(f"{len(steps)+1}. Sử dụng LLM để tổng hợp và sinh câu trả lời")
        else:
            steps.append(f"{len(steps)+1}. Tổng hợp và sinh câu trả lời")
        return steps

    def _compute_confidence(self, context: AugmentedContext) -> float:
        """Tính confidence score dựa trên retrieval scores."""
        if self.generation_mode == "llm":
            return 0.85
        if not context.documents:
            return 0.0
        scores = [d.score for d in context.documents[:3]]
        avg_score = sum(scores) / len(scores)
        # Normalize to 0-1
        confidence = min(avg_score / max(scores), 1.0) if max(scores) > 0 else 0.5
        return round(confidence, 3)
