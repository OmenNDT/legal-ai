"""Knowledge Graph: Entity and relation extraction from Vietnamese legal text.

Extracts 9 entity types and 7 relation types from legal documents.
Uses regex for structural extraction + NER for semantic extraction.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.common.config import ENTITY_TYPES, RELATION_TYPES, DOCUMENT_TYPES
from src.common.text_processor import extract_legal_references


@dataclass
class LegalEntity:
    """A node in the legal knowledge graph."""
    id: str                        # Unique identifier (e.g., "TT_99_2015")
    type: str                       # Entity type (LUAT, THONG_TU, DIEU, etc.)
    name: str                       # Full name (e.g., "Thông tư 99/2015/TT-BTC")
    properties: dict = field(default_factory=dict)  # Additional attributes


@dataclass
class LegalRelation:
    """An edge in the legal knowledge graph."""
    source_id: str    # Source entity ID
    target_id: str    # Target entity ID
    relation_type: str  # One of RELATION_TYPES
    properties: dict = field(default_factory=dict)


class LegalEntityExtractor:
    """Extract legal entities from text using regex patterns.

    9 entity types: LUAT, THONG_TU, NGHI_DINH, DIEU, KHOAN, DIEM,
                    CO_QUAN, KHAISUAT, NGAY_THANG
    """

    PATTERNS = {
        "THONG_TU": [
            r"Thông\s+tư\s+(?:số\s+)?(\d+(?:/\d+)*(?:/TT-\w+)?)",
            r"TT\s+(?:số\s+)?(\d+(?:/\d+)*)",
        ],
        "NGHI_DINH": [
            r"Nghị\s+định\s+(?:số\s+)?(\d+(?:/\d+)*(?:/NĐ-CP)?)",
            r"NĐ\s+(?:số\s+)?(\d+(?:/\d+)*)",
        ],
        "LUAT": [
            r"Luật\s+([\w\s]+?)(?:\s+số|năm|\d{4}|$)",
            r"Bộ\s+luật\s+([\w\s]+?)(?:\s+số|năm|\d{4}|$)",
        ],
        "DIEU": [
            r"Điều\s+(\d+)",
            r"điều\s+(\d+)",
        ],
        "KHOAN": [
            r"Khoản\s+(\d+)",
            r"khoản\s+(\d+)",
        ],
        "DIEM": [
            r"Điểm\s+([a-zđ]+)",
            r"điểm\s+([a-zđ]+)",
        ],
        "CO_QUAN": [
            r"Bộ\s+([\w\s]+?)(?:\s*,|\s*\.|$)",
            r"Ủy\s+ban\s+([\w\s]+?)(?:\s*,|\s*\.|$)",
            r"Chính\s+phủ",
            r"Quốc\s+hội",
            r"Hội\s+đồng\s+([\w\s]+?)(?:\s*,|\s*\.|$)",
        ],
        "NGAY_THANG": [
            r"ngày\s+(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
            r"ngày\s+(\d{1,2}/\d{1,2}/\d{4})",
        ],
    }

    def extract(self, text: str) -> list[LegalEntity]:
        """Extract all legal entities from text."""
        entities = []
        seen_ids = set()

        # Use common extractor first (handles most structural refs)
        refs = extract_legal_references(text)
        for ref in refs:
            entity_id = f"{ref['type']}_{ref['value'].replace('/', '_').replace(' ', '_')}"
            if entity_id not in seen_ids:
                seen_ids.add(entity_id)
                entities.append(LegalEntity(
                    id=entity_id,
                    type=ref["type"],
                    name=ref["full"],
                    properties={"value": ref["value"], "position": ref["start"]},
                ))

        # Additional patterns not covered by common extractor
        for entity_type, patterns in self.PATTERNS.items():
            if entity_type in ("THONG_TU", "NGHI_DINH", "LUAT", "DIEU", "KHOAN", "DIEM"):
                continue  # Already handled by extract_legal_references

            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_id = f"{entity_type}_{match.group(0).replace(' ', '_')}"
                    if entity_id not in seen_ids:
                        seen_ids.add(entity_id)
                        entities.append(LegalEntity(
                            id=entity_id,
                            type=entity_type,
                            name=match.group(0),
                            properties={"position": match.start()},
                        ))

        return entities


class LegalRelationExtractor:
    """Extract legal relations from text.

    7 relation types: DUA_TREN, THAM_CHIEU, HET_HIEU_LUC, THAY_THE,
                      SUA_DOI_BO_SUNG, HUONG_DAN, CHUA
    """

    RELATION_PATTERNS = {
        "DUA_TREN": [
            r"căn cứ\s+([\w\s]+?Luật[\w\s]+)",
            "căn cứ\s+(.+?)(?:\s*[;,]\s*căn cứ)",
            r"dựa\s+trên\s+(.+?)(?:\s*[;,.])",
        ],
        "THAM_CHIEU": [
            r"tham\s+chiếu\s+(?:điều\s+\d+.*?)?(?:của\s+)?([\w\s]+?Luật[\w\s]+|Thông\s+tư\s+[\d/]+|Nghị\s+định\s+[\d/]+)",
            r"theo\s+([\w\s]+?Luật[\w\s]+|Thông\s+tư\s+[\d/]+|Nghị\s+định\s+[\d/]+)",
        ],
        "HET_HIEU_LUC": [
            r"hết\s+hiệu\s+lực",
            r"bãi\s+bỏ",
            r"thu\s+hồi",
        ],
        "THAY_THE": [
            r"thay\s+thế\s+(.+?)(?:\s*[;.])",
            r"thay\s+cho\s+(.+?)(?:\s*[;.])",
        ],
        "SUA_DOI_BO_SUNG": [
            r"sửa\s+đổi|bổ\s+sung",
            r"sửa\s+đổi,\s*bổ\s+sung",
        ],
        "HUONG_DAN": [
            r"hướng\s+dẫn\s+thi\s+hành",
            r"hướng\s+dẫn\s+nghiệp\s+vụ",
        ],
        "CHUA": [
            r"gồm\s+\d+\s+(?:điều|khoản)",
            r"được\s+chia\s+thành\s+\d+\s+(?:chương|phần|mục)",
        ],
    }

    def extract(self, text: str, doc_entity: LegalEntity) -> list[LegalRelation]:
        """Extract relations from a legal document text.

        Args:
            text: Full document text
            doc_entity: The document entity (source of relations)

        Returns:
            List of LegalRelation objects
        """
        relations = []
        entity_extractor = LegalEntityExtractor()
        entities = entity_extractor.extract(text)

        for rel_type, patterns in self.RELATION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Find target entities near the match
                    match_start = max(0, match.start() - 50)
                    match_end = min(len(text), match.end() + 200)
                    context = text[match_start:match_end]

                    # Look for referenced documents in context
                    target_refs = extract_legal_references(context)
                    for ref in target_refs:
                        target_id = f"{ref['type']}_{ref['value'].replace('/', '_').replace(' ', '_')}"
                        if target_id != doc_entity.id:
                            relations.append(LegalRelation(
                                source_id=doc_entity.id,
                                target_id=target_id,
                                relation_type=rel_type,
                                properties={
                                    "evidence": match.group(0),
                                    "position": match.start(),
                                },
                            ))

        # Structural relations: document contains articles
        for entity in entities:
            if entity.type == "DIEU" and entity.id != doc_entity.id:
                relations.append(LegalRelation(
                    source_id=doc_entity.id,
                    target_id=entity.id,
                    relation_type="CHUA",
                    properties={"article": entity.properties.get("value", "")},
                ))

        return relations


def determine_document_type(title: str, content: str) -> str:
    """Classify a legal document into one of DOCUMENT_TYPES."""
    text = (title + " " + content[:500]).lower()

    type_keywords = {
        "hien_phap": ["hiến pháp"],
        "bo_luat": ["bộ luật", "bộ luật"],
        "luat": ["luật ", "luật số"],
        "nghi_dinh": ["nghị định", "nghị quyết số", "nđ-cp"],
        "thong_tu": ["thông tư", "tt-btc", "tt-bnn", "tt-byt"],
        "quyet_dinh": ["quyết định"],
        "nghi_quyet": ["nghị quyết"],
    }

    for doc_type, keywords in type_keywords.items():
        for kw in keywords:
            if kw in text:
                return doc_type

    return "other"