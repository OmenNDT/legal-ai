import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

class MetadataTagger:
    _DOMAIN_MAP = {
        "lao_dong": ["lao động", "hợp đồng lao động", "tiền lương", "bảo hiểm xã hội", "sa thải", "thôi việc", "người lao động"],
        "dan_su": ["dân sự", "hợp đồng", "tài sản", "thừa kế", "ly hôn", "hôn nhân gia đình", "bồi thường"],
        "hinh_su": ["hình sự", "tội phạm", "xử phạt", "phạt tù", "tội", "vi phạm hình sự", "truy tố"],
        "hanh_chinh": ["hành chính", "xử phạt hành chính", "giấy phép", "thủ tục hành chính", "cơ quan nhà nước"],
        "thuong_mai": ["thương mại", "doanh nghiệp", "kinh doanh", "thuế", "xuất nhập khẩu", "hải quan"],
        "dat_dai": ["đất đai", "bất động sản", "quyền sử dụng đất", "nhà ở", "xây dựng"],
        "ke_toan": ["kế toán", "tài chính", "báo cáo tài chính", "kiểm toán", "ngân sách"],
    }
    _STOP_WORDS = {
        "và", "của", "trong", "là", "có", "các", "được", "theo", "cho", "với",
        "về", "đến", "từ", "tại", "hoặc", "nhưng", "nếu", "khi", "này", "đó",
        "để", "không", "thì", "mà", "như", "phải", "sẽ", "đã", "vì", "bởi",
    }
    _KEYWORD_MIN_LEN = 3
    _MAX_KEYWORDS = 10

    def __init__(self, db):
        self._db = db

    def tag(self, chunks: list) -> None:
        logger.info("[Tag] Input: %d chunk(s)", len(chunks))
        tags_batch = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk.get("chunk_text", "")
            doc_type = chunk.get("document_type", "")
            effective_status = chunk.get("effective_status", "active")

            domain = self._detect_domain(text, doc_type)
            keywords = self._extract_keywords(text)

            tags_batch.append((chunk_id, "domain", domain))
            tags_batch.append((chunk_id, "document_type", doc_type))
            tags_batch.append((chunk_id, "effective_status", effective_status))
            for kw in keywords:
                tags_batch.append((chunk_id, "keyword", kw))

        if tags_batch:
            try:
                self._db.execute_many(
                    "INSERT INTO legal_metadata_tags (chunk_id, tag_key, tag_value) VALUES (%s, %s, %s)",
                    tags_batch,
                )
                logger.info("[Tag] Inserted %d tag(s)", len(tags_batch))
            except Exception as e:
                logger.error("[Tag] DB insert FAILED: %s", e, exc_info=True)

    def _detect_domain(self, text: str, document_type: str) -> str:
        lower_text = text.lower()
        domain_scores: dict = {}
        for domain, keywords in self._DOMAIN_MAP.items():
            score = sum(1 for kw in keywords if kw in lower_text)
            if score > 0:
                domain_scores[domain] = score
        if domain_scores:
            return max(domain_scores, key=lambda d: domain_scores[d])
        return "general"

    def _extract_keywords(self, text: str) -> list:
        words = re.findall(r"[a-zA-ZÀ-ỹ]{3,}", text.lower())
        filtered = [w for w in words if w not in self._STOP_WORDS and len(w) >= self._KEYWORD_MIN_LEN]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(self._MAX_KEYWORDS)]
