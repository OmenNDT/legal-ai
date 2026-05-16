"""Query expansion with legal synonyms and domain detection."""

from src.rag_pipeline.contracts import ExtractedEntity

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


def expand_query(query: str, entities: list[ExtractedEntity]) -> list[str]:
    """Expand query with legal synonyms and entity names."""
    queries = [query]
    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in query:
            for syn in synonyms:
                queries.append(query.replace(term, syn))
    entity_names = [e.text for e in entities if e.label in ("LUAT", "THONG_TU", "NGHI_DINH")]
    if entity_names:
        queries.append(f"{query} {' '.join(entity_names)}")
    seen = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped[:5]


def detect_domain(query: str) -> str:
    """Detect legal domain from query keywords."""
    for domain, keywords in DOMAIN_MAP.items():
        for kw in keywords:
            if kw in query:
                return domain
    return "general"
