"""Mock adapters cho Phần 1 (Preprocessing) và Phần 5 (Postprocessing).

Dùng để test pipeline 2→3→4 độc lập mà không cần các module khác.
"""

import re
from src.rag_pipeline.contracts import (
    ProcessedQuestion,
    ExtractedEntity,
    GeneratedAnswer,
)


class MockPreprocessor:
    """Giả lập Phần 1: Tiền xử lý câu hỏi.

    Trong thực tế, phần này do nhóm bạn code (React frontend + NLP preprocessing).
    Mock này cung cấp đủ dữ liệu để test pipeline 2→3→4.
    """

    # Từ khóa để nhận diện intent đơn giản
    INTENT_KEYWORDS = {
        "hoi_dieu_khoan": ["điều", "khoản", "quy định"],
        "hoi_hieu_luc": ["hiệu lực", "còn hiệu lực", "hết hiệu lực"],
        "hoi_sua_doi": ["sửa đổi", "thay thế", "bãi bỏ"],
        "hoi_dinh_nghia": ["là gì", "định nghĩa", "khái niệm"],
        "hoi_thu_tuc": ["thủ tục", "quy trình", "làm thế nào"],
        "hoi_hinh_phat": ["hình phạt", "xử phạt", "chế tài", "phạt"],
        "hoi_quyen_loi": ["quyền", "lợi ích", "được hưởng"],
        "hoi_nghia_vu": ["nghĩa vụ", "bắt buộc", "phải"],
        "hoi_thoi_han": ["thời hạn", "thời hiệu", "thời gian"],
        "tra_cuu_luat": ["luật", "bộ luật"],
        "tra_cuu_nghi_dinh": ["nghị định"],
        "tra_cuu_thong_tu": ["thông tư"],
        "hoi_tom_tat": ["tóm tắt", "tổng quan"],
        "hoi_tong_hop": ["tổng hợp", "tất cả"],
    }

    # Pattern nhận diện entity
    ENTITY_PATTERNS = {
        "LUAT": r"Luật\s+[A-ZÀ-Ỹ][a-zà-ỹ\s]+(?:\d{4})?",
        "NGHI_DINH": r"Nghị\s+định\s+(?:số\s+)?\d+/\d{4}/NĐ-CP",
        "THONG_TU": r"Thông\s+tư\s+(?:số\s+)?\d+/\d{4}/(?:TT-BTC|TT-BLĐTBXH|TT-BTP|TT-BTNMT)",
        "DIEU": r"Điều\s+\d+(?:\.\d+)?",
        "KHOAN": r"khoản\s+\d+",
        "DIEM": r"điểm\s+[a-zđ]",
        "NGAY_THANG": r"\d{1,2}/\d{1,2}/\d{4}",
    }

    def process(self, raw_question: str) -> ProcessedQuestion:
        """Tiền xử lý câu hỏi và trả về ProcessedQuestion."""
        # Simple word segmentation (mock)
        segmented = self._mock_segment(raw_question)

        # Intent classification
        intent, confidence = self._mock_classify_intent(raw_question)

        # Entity extraction
        entities = self._mock_extract_entities(raw_question)

        # Build filters from entities
        filters = self._build_filters(entities)

        return ProcessedQuestion(
            raw_text=raw_question,
            segmented_text=segmented,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            filters=filters,
        )

    def _mock_segment(self, text: str) -> str:
        """Mock word segmentation: giữ nguyên hoặc thay space bằng underscore cho từ ghép."""
        # Đơn giản: giữ nguyên text
        return text

    def _mock_classify_intent(self, text: str) -> tuple[str, float]:
        """Mock intent classification dựa trên keyword matching."""
        text_lower = text.lower()
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score

        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(0.5 + scores[best_intent] * 0.1, 0.95)
            return best_intent, round(confidence, 2)

        return "hoi_tong_hop", 0.5  # Default fallback

    def _mock_extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Mock NER dựa trên regex patterns."""
        entities = []
        for label, pattern in self.ENTITY_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                ))
        return entities

    def _build_filters(self, entities: list[ExtractedEntity]) -> dict:
        """Build metadata filters từ entities."""
        filters = {}
        for ent in entities:
            if ent.label == "LUAT":
                filters["doc_type"] = "law"  # Match actual data
            elif ent.label == "NGHI_DINH":
                filters["doc_type"] = "decree"
            elif ent.label == "THONG_TU":
                filters["doc_type"] = "circular"
        return filters


class MockPostprocessor:
    """Giả lập Phần 5: Hậu xử lý & trả kết quả.

    Trong thực tế, phần này do nhóm bạn code (format Markdown/HTML, lưu lịch sử).
    Mock này cung cấp output đơn giản để test.
    """

    def format(self, answer: GeneratedAnswer, return_markdown: bool = True) -> dict:
        """Format câu trả lời thành response cuối cùng."""
        result = {
            "answer": answer.answer_text,
            "confidence": answer.confidence,
            "reasoning": answer.reasoning_steps,
            "generation_method": answer.generation_method,
            "latency_ms": answer.latency_ms,
        }

        if answer.citations:
            result["sources"] = [
                {
                    "doc_id": c.doc_id,
                    "name": c.doc_name,
                    "excerpt": c.excerpt[:150] + "..." if len(c.excerpt) > 150 else c.excerpt,
                    "relevance": round(c.relevance_score, 3),
                }
                for c in answer.citations
            ]

        if return_markdown:
            result["markdown"] = self._to_markdown(answer)

        return result

    def _to_markdown(self, answer: GeneratedAnswer) -> str:
        """Convert answer to Markdown format."""
        md = f"## Câu trả lời\n\n{answer.answer_text}\n\n"

        if answer.citations:
            md += "### Nguồn tham khảo\n\n"
            for i, c in enumerate(answer.citations, 1):
                md += f"{i}. **{c.doc_name}** (ID: {c.doc_id})\n"
                md += f"   - Đoạn trích: {c.excerpt[:100]}...\n"
                md += f"   - Độ liên quan: {c.relevance_score:.3f}\n\n"

        if answer.reasoning_steps:
            md += "### Quá trình suy luận\n\n"
            for step in answer.reasoning_steps:
                md += f"- {step}\n"

        md += f"\n---\n*Độ tin cậy: {answer.confidence:.1%} | "
        md += f"Phương pháp: {answer.generation_method} | "
        md += f"Thời gian xử lý: {answer.latency_ms:.0f}ms*"

        return md
