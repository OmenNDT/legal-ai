from dataclasses import dataclass, field
from typing import Optional
import re

from src.common.text_processor import clean_vietnamese

DOMAIN_MAP = {
    "lao_dong": ["lao động", "hợp đồng lao động", "tiền lương", "bảo hiểm xã hội", "sa thải", "thôi việc"],
    "dan_su": ["dân sự", "hợp đồng", "tài sản", "thừa kế", "ly hôn", "hôn nhân gia đình"],
    "hinh_su": ["hình sự", "tội phạm", "xử phạt", "phạt tù", "tội", "vi phạm hình sự"],
    "hanh_chinh": ["hành chính", "xử phạt hành chính", "giấy phép", "thủ tục hành chính"],
    "thuong_mai": ["thương mại", "doanh nghiệp", "kinh doanh", "thuế", "xuất nhập khẩu"],
    "dat_dai": ["đất đai", "bất động sản", "quyền sử dụng đất", "nhà ở"],
    "ke_toan": ["kế toán", "tài chính", "báo cáo tài chính", "kiểm toán"],
}

LEGAL_SYNONYMS = {
    "sa thải": ["chấm dứt hợp đồng lao động", "đuổi việc"],
    "phạt": ["xử phạt", "chế tài", "hình phạt"],
    "hợp đồng": ["thỏa thuận", "giao kết"],
    "hiệu lực": ["còn hiệu lực", "có giá trị pháp lý"],
    "điều khoản": ["quy định", "nội dung"],
}

@dataclass
class StructuredQuery:
    raw_query: str
    normalized_query: str
    intent: str
    intent_confidence: float
    entities: list
    domain: str
    expanded_queries: list
    rewritten_query: str
    query_id: Optional[int] = None


class QueryUnderstanding:
    def __init__(self, intent_classifier, ner_tagger, intent_tokenizer, ner_tokenizer, llm_client=None):
        self._intent_classifier = intent_classifier
        self._ner_tagger = ner_tagger
        self._intent_tokenizer = intent_tokenizer
        self._ner_tokenizer = ner_tokenizer
        self._llm_client = llm_client

    def process(self, raw_query: str) -> StructuredQuery:
        normalized = self._normalize(raw_query)
        intent_result = self._intent_classifier.predict(normalized, self._intent_tokenizer)
        entities = self._ner_tagger.predict(normalized, self._ner_tokenizer)
        domain = self._detect_domain(normalized, entities)
        expanded = self._expand_query(normalized, entities)
        rewritten = self._rewrite_query(normalized, intent_result["intent"], entities, domain)

        return StructuredQuery(
            raw_query=raw_query,
            normalized_query=normalized,
            intent=intent_result["intent"],
            intent_confidence=intent_result["confidence"],
            entities=entities,
            domain=domain,
            expanded_queries=expanded,
            rewritten_query=rewritten,
        )

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        return clean_vietnamese(text)

    def _detect_domain(self, query: str, entities: list) -> str:
        for domain, keywords in DOMAIN_MAP.items():
            for kw in keywords:
                if kw in query:
                    return domain
        return "general"

    def _expand_query(self, query: str, entities: list) -> list:
        expanded = [query]
        for term, synonyms in LEGAL_SYNONYMS.items():
            if term in query:
                for syn in synonyms:
                    expanded.append(query.replace(term, syn))
        entity_terms = [e["text"] for e in entities if e["entity"] in ("LUAT", "THONG_TU", "NGHI_DINH")]
        if entity_terms:
            expanded.append(f"{query} {' '.join(entity_terms)}")
        return list(dict.fromkeys(expanded))[:5]

    def _rewrite_query(self, query: str, intent: str, entities: list, domain: str) -> str:
        if self._llm_client is None:
            return self._rule_based_rewrite(query, entities)

        prompt = (
            f"Viết lại câu hỏi pháp lý sau thành truy vấn tìm kiếm văn bản luật ngắn gọn, rõ ràng, giữ nguyên ý nghĩa.\n"
            f"Lĩnh vực: {domain}\n"
            f"Câu hỏi: {query}\n"
            f"Câu hỏi viết lại (chỉ trả về câu hỏi, không giải thích):"
        )
        return self._llm_client.complete(prompt, max_tokens=80)

    def _rule_based_rewrite(self, query: str, entities: list) -> str:
        entity_refs = " ".join(
            f"{e['entity']} {e['text']}"
            for e in entities
            if e["entity"] in ("LUAT", "THONG_TU", "NGHI_DINH", "DIEU")
        )
        return f"{query} {entity_refs}".strip() if entity_refs else query
