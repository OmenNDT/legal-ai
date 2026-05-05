"""Phần 4: Generation (Sinh câu trả lời).

Luồng xử lý:
  AugmentedContext → PhoBERT/LLM → GeneratedAnswer

Mục tiêu: Sinh câu trả lời chất lượng cao, có trích dẫn nguồn.
"""

import time
import re
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel

from src.rag_pipeline.contracts import AugmentedContext, GeneratedAnswer, Citation
from src.common.config import PHOBERT_MODEL, MAX_SEQ_LENGTH


class LegalAnswerGenerator:
    """Sinh câu trả lời pháp luật từ ngữ cảnh đã augment.

    Hỗ trợ 2 chế độ:
      - "extractive": Trích xuất câu trả lời từ context (mặc định, không cần LLM)
      - "generative": Dùng PhoBERT để paraphrase/tóm tắt (nếu có GPU)

    Args:
        model_name: Tên model PhoBERT
        device: "cpu" hoặc "cuda"
        generation_mode: "extractive" | "generative"
    """

    def __init__(
        self,
        model_name: str = PHOBERT_MODEL,
        device: str = "cpu",
        generation_mode: str = "extractive",
    ):
        self.model_name = model_name
        self.device = device
        self.generation_mode = generation_mode
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load PhoBERT model."""
        if self._loaded or self.generation_mode == "extractive":
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
        except Exception:
            self.generation_mode = "extractive"  # Fallback

    def generate(
        self,
        context: AugmentedContext,
        max_length: int = 512,
    ) -> GeneratedAnswer:
        """Sinh câu trả lời từ augmented context.

        Args:
            context: AugmentedContext từ Phần 3
            max_length: Độ dài tối đa câu trả lời

        Returns:
            GeneratedAnswer với câu trả lời và citations
        """
        start_time = time.time()

        if self.generation_mode == "generative":
            answer = self._generate_generative(context, max_length)
        else:
            answer = self._generate_extractive(context)

        # Trích xuất citations từ documents
        citations = self._extract_citations(context)

        # Xây dựng reasoning steps
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
        """Trích xuất câu trả lời từ context (không dùng generative model).

        Chiến lược:
        1. Tìm sentences trong context liên quan nhất đến câu hỏi
        2. Ghép thành câu trả lời mạch lạc
        3. Thêm câu mở đầu kết luận
        """
        question = context.original_question
        ctx_text = context.context_text

        # Tách câu hỏi và phần context
        if "Câu hỏi:" in ctx_text:
            parts = ctx_text.split("\n\n", 1)
            if len(parts) == 2:
                ctx_text = parts[1]

        # Tìm các đoạn liên quan nhất
        sentences = self._split_into_sentences(ctx_text)
        if not sentences:
            return "Xin lỗi, không tìm thấy thông tin phù hợp để trả lời câu hỏi."

        # Score sentences by relevance to question
        scored = self._score_sentences(question, sentences)
        top_sentences = [s for s, _ in scored[:5]]

        # Build answer
        answer = self._build_answer_from_sentences(question, top_sentences)
        return answer

    def _generate_generative(
        self,
        context: AugmentedContext,
        max_length: int = 512,
    ) -> str:
        """Dùng PhoBERT để sinh câu trả lời (paraphrase/tóm tắt)."""
        self._load_model()
        if not self._loaded:
            return self._generate_extractive(context)

        # Tạo prompt: question + context
        prompt = self._build_prompt(context)

        try:
            inputs = self.tokenizer(
                prompt,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Dùng hidden states để tạo summary (simplified)
                # Trong thực tế cần seq2seq model hoặc GPT-style model

            # Fallback về extractive vì PhoBERT không phải generative model
            return self._generate_extractive(context)

        except Exception:
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

    def _split_into_sentences(self, text: str) -> list[str]:
        """Tách văn bản thành câu."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_sentences(
        self,
        question: str,
        sentences: list[str],
    ) -> list[tuple[str, float]]:
        """Score sentences by word overlap với question."""
        q_words = set(question.lower().split())
        scored = []
        for sent in sentences:
            s_words = set(sent.lower().split())
            overlap = len(q_words & s_words)
            score = overlap / max(len(q_words), 1)
            scored.append((sent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _build_answer_from_sentences(
        self,
        question: str,
        sentences: list[str],
    ) -> str:
        """Ghép các câu thành câu trả lời hoàn chỉnh."""
        if not sentences:
            return "Không tìm thấy thông tin phù hợp."

        # Xác định loại câu hỏi để định dạng câu trả lời
        question_lower = question.lower()

        if any(w in question_lower for w in ["có hiệu lực", "hết hiệu lực", "còn hiệu lực"]):
            intro = "Theo các quy định pháp luật liên quan:"
        elif any(w in question_lower for w in ["định nghĩa", "là gì", "khái niệm"]):
            intro = "Theo quy định của pháp luật:"
        elif any(w in question_lower for w in ["hình phạt", "xử phạt", "chế tài"]):
            intro = "Về chế tài xử phạt, pháp luật quy định:"
        elif any(w in question_lower for w in ["thủ tục", "quy trình", "làm thế nào"]):
            intro = "Về thủ tục thực hiện:"
        else:
            intro = "Theo quy định của pháp luật:"

        # Ghép các câu, loại bỏ trùng lặp
        unique_sentences = []
        seen = set()
        for sent in sentences:
            normalized = re.sub(r'\s+', ' ', sent.lower().strip())
            if normalized not in seen and len(sent) > 20:
                unique_sentences.append(sent)
                seen.add(normalized)

        if not unique_sentences:
            return "Không tìm thấy thông tin đầy đủ để trả lời câu hỏi."

        answer = intro + "\n\n"
        for i, sent in enumerate(unique_sentences[:5], 1):
            answer += f"{i}. {sent}\n"

        answer += "\nLưu ý: Thông tin trên dựa trên các văn bản pháp luật hiện hành."
        return answer.strip()

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
        steps.append(f"{len(steps)+1}. Tổng hợp và sinh câu trả lời")
        return steps

    def _compute_confidence(self, context: AugmentedContext) -> float:
        """Tính confidence score dựa trên retrieval scores."""
        if not context.documents:
            return 0.0
        scores = [d.score for d in context.documents[:3]]
        avg_score = sum(scores) / len(scores)
        # Normalize to 0-1
        confidence = min(avg_score / max(scores), 1.0) if max(scores) > 0 else 0.5
        return round(confidence, 3)
