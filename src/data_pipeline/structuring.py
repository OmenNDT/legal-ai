import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

from src.knowledge.entity_extractor import determine_document_type

@dataclass
class ParsedMetadata:
    title: str
    document_number: str
    document_type: str
    issuing_body: str
    effective_date: Optional[date]
    expiry_date: Optional[date]

@dataclass
class ParsedSection:
    article_number: str
    clause_number: str
    point_number: str
    title: str
    content: str
    level: int
    parent_key: Optional[str]

class LegalMetadataParser:
    _DOC_NUMBER_RE = re.compile(r"Số[:\s]+([0-9][^\n]{0,60})", re.IGNORECASE)
    _DATE_RE = re.compile(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE
    )
    _EXPIRY_RE = re.compile(
        r"hết\s+hiệu\s+lực[^.]{0,60}ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        re.IGNORECASE,
    )
    _ISSUING_BODIES = [
        "QUỐC HỘI", "ỦY BAN THƯỜNG VỤ QUỐC HỘI", "CHỦ TỊCH NƯỚC",
        "CHÍNH PHỦ", "THỦ TƯỚNG CHÍNH PHỦ", "NGÂN HÀNG NHÀ NƯỚC",
        "BỘ TÀI CHÍNH", "BỘ TƯ PHÁP", "BỘ LAO ĐỘNG", "BỘ CÔNG THƯƠNG",
        "BỘ Y TẾ", "BỘ GIÁO DỤC", "BỘ CÔNG AN", "BỘ QUỐC PHÒNG",
        "BỘ KẾ HOẠCH VÀ ĐẦU TƯ", "BỘ TÀI NGUYÊN VÀ MÔI TRƯỜNG",
    ]

    def parse(self, text: str, filename: str) -> ParsedMetadata:
        title = self._extract_title(text, filename)
        doc_number = self._extract_doc_number(text)
        doc_type = determine_document_type(title, text) or "other"
        issuing_body = self._extract_issuing_body(text)
        effective_date = self._extract_date(text, self._DATE_RE)
        expiry_date = self._extract_date(text, self._EXPIRY_RE)
        return ParsedMetadata(
            title=title,
            document_number=doc_number,
            document_type=doc_type,
            issuing_body=issuing_body,
            effective_date=effective_date,
            expiry_date=expiry_date,
        )

    def _extract_title(self, text: str, filename: str) -> str:
        skip_re = re.compile(
            r"^(Số|www|http|Copyright|©|\d{1,4}$|Độc\s+lập|Cộng\s+hòa)",
            re.IGNORECASE,
        )
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 15 and not skip_re.match(line):
                return line[:500]
        return filename

    def _extract_doc_number(self, text: str) -> str:
        m = self._DOC_NUMBER_RE.search(text[:3000])
        if m:
            return m.group(1).strip()[:50]
        return ""

    def _extract_issuing_body(self, text: str) -> str:
        upper_text = text[:3000].upper()
        for body in self._ISSUING_BODIES:
            if body in upper_text:
                return body
        return ""

    def _extract_date(self, text: str, pattern: re.Pattern) -> Optional[date]:
        m = pattern.search(text[:5000])
        if m:
            try:
                return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except (ValueError, IndexError):
                pass
        return None

class SectionParser:
    _ARTICLE_RE = re.compile(r"^Điều\s+(\d+)[\.:]?\s*(.*)", re.IGNORECASE | re.MULTILINE)
    _CLAUSE_RE = re.compile(r"^\s*(\d+)\.\s+(.*)", re.MULTILINE)
    _POINT_RE = re.compile(r"^\s*([a-zđ])\)\s+(.*)", re.MULTILINE)

    def parse(self, text: str) -> list:
        articles = list(self._ARTICLE_RE.finditer(text))
        if not articles:
            return self._fallback_section(text)

        sections = []
        for i, article_match in enumerate(articles):
            article_num = article_match.group(1)
            article_title = article_match.group(2).strip()
            body_start = article_match.end()
            body_end = articles[i + 1].start() if i + 1 < len(articles) else len(text)
            article_body = text[body_start:body_end].strip()

            clauses = list(self._CLAUSE_RE.finditer(article_body))
            if clauses:
                sections.extend(self._parse_clauses(article_body, article_num, clauses))
            else:
                sections.append(ParsedSection(
                    article_number=article_num,
                    clause_number="",
                    point_number="",
                    title=article_title,
                    content=article_body,
                    level=1,
                    parent_key=None,
                ))
        return sections

    def _fallback_section(self, text: str) -> list:
        if not text.strip():
            return []
        return [ParsedSection(
            article_number="", clause_number="", point_number="",
            title="Nội dung", content=text.strip(), level=1, parent_key=None,
        )]

    def _parse_clauses(self, article_body: str, article_num: str, clauses: list) -> list:
        sections = []
        for i, clause_match in enumerate(clauses):
            clause_num = clause_match.group(1)
            body_start = clause_match.end()
            body_end = clauses[i + 1].start() if i + 1 < len(clauses) else len(article_body)
            clause_body = (clause_match.group(2) + " " + article_body[body_start:body_end]).strip()

            points = list(self._POINT_RE.finditer(clause_body))
            if points:
                sections.extend(self._parse_points(clause_body, article_num, clause_num, points))
            else:
                sections.append(ParsedSection(
                    article_number=article_num,
                    clause_number=clause_num,
                    point_number="",
                    title="",
                    content=clause_body,
                    level=2,
                    parent_key=f"dieu_{article_num}",
                ))
        return sections

    def _parse_points(self, clause_body: str, article_num: str, clause_num: str, points: list) -> list:
        sections = []
        for i, point_match in enumerate(points):
            point_letter = point_match.group(1)
            body_start = point_match.end()
            body_end = points[i + 1].start() if i + 1 < len(points) else len(clause_body)
            point_body = (point_match.group(2) + " " + clause_body[body_start:body_end]).strip()
            sections.append(ParsedSection(
                article_number=article_num,
                clause_number=clause_num,
                point_number=point_letter,
                title="",
                content=point_body,
                level=3,
                parent_key=f"dieu_{article_num}_khoan_{clause_num}",
            ))
        return sections

class DocumentStructurer:
    def __init__(self, db):
        self._db = db
        self._metadata_parser = LegalMetadataParser()
        self._section_parser = SectionParser()

    def structure(self, cleaned_docs: list) -> list:
        logger.info("[Structure] Input: %d doc(s)", len(cleaned_docs))
        results = []
        for doc in cleaned_docs:
            try:
                result = self._process_doc(doc)
                if result:
                    results.append(result)
                    logger.info(
                        "[Structure] OK: %s — version_id=%s, sections=%d",
                        doc.get("filename", "?"),
                        result.get("version_id"),
                        len(result.get("sections", [])),
                    )
                else:
                    logger.warning("[Structure] Skipped (empty): %s", doc.get("filename", "?"))
            except Exception as e:
                logger.error("[Structure] FAILED: %s — %s", doc.get("filename", "?"), e, exc_info=True)
        logger.info("[Structure] Done: %d doc(s)", len(results))
        return results

    def _process_doc(self, doc: dict) -> Optional[dict]:
        text = doc.get("clean_text") or doc.get("raw_text", "")
        if not text.strip():
            return None
        filename = doc.get("filename", "")
        metadata = self._metadata_parser.parse(text, filename)
        logger.info(
            "[Structure] Metadata: title=%r, doc_number=%r, type=%s, body=%s",
            metadata.title[:60],
            metadata.document_number,
            metadata.document_type,
            metadata.issuing_body,
        )
        document_id = self._upsert_document(metadata)
        self._supersede_active_versions(document_id)
        version_id = self._insert_version(document_id, metadata, text)
        raw_sections = self._section_parser.parse(text)
        logger.info("[Structure] Sections parsed: %d", len(raw_sections))
        section_rows = self._insert_sections(version_id, raw_sections)
        return {
            **doc,
            "document_id": document_id,
            "version_id": version_id,
            "sections": section_rows,
            "effective_status": "active",
            "document_type": metadata.document_type,
            "document_title": metadata.title,
        }

    def _upsert_document(self, metadata: ParsedMetadata) -> int:
        if metadata.document_number:
            rows = self._db.execute_query(
                "SELECT id FROM legal_documents WHERE document_number = %s",
                (metadata.document_number,),
            )
            if rows:
                return rows[0][0]
        rows = self._db.execute_query(
            "INSERT INTO legal_documents (title, document_number, document_type, issuing_body) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (metadata.title, metadata.document_number, metadata.document_type, metadata.issuing_body),
        )
        return rows[0][0]

    def _supersede_active_versions(self, document_id: int) -> None:
        self._db.execute_query(
            "UPDATE document_versions SET status = 'replaced', expiry_date = CURRENT_DATE "
            "WHERE document_id = %s AND status = 'active'",
            (document_id,),
            fetch=False,
        )

    def _insert_version(self, document_id: int, metadata: ParsedMetadata, raw_text: str) -> int:
        rows = self._db.execute_query(
            "SELECT COALESCE(MAX(version_number), 0) + 1 FROM document_versions WHERE document_id = %s",
            (document_id,),
        )
        version_number = rows[0][0] if rows else 1
        rows = self._db.execute_query(
            "INSERT INTO document_versions "
            "(document_id, version_number, effective_date, expiry_date, status, raw_text) "
            "VALUES (%s, %s, %s, %s, 'active', %s) RETURNING id",
            (document_id, version_number, metadata.effective_date, metadata.expiry_date, raw_text),
        )
        return rows[0][0]

    def _insert_sections(self, version_id: int, sections: list) -> list:
        key_to_id: dict = {}
        section_rows = []
        for section in sections:
            parent_id = key_to_id.get(section.parent_key) if section.parent_key else None
            rows = self._db.execute_query(
                "INSERT INTO legal_sections "
                "(version_id, article_number, clause_number, point, title, content, parent_id, level) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    version_id,
                    section.article_number,
                    section.clause_number,
                    section.point_number,
                    section.title,
                    section.content,
                    parent_id,
                    section.level,
                ),
            )
            section_id = rows[0][0]
            if section.article_number and not section.clause_number:
                key_to_id[f"dieu_{section.article_number}"] = section_id
            elif section.article_number and section.clause_number and not section.point_number:
                key_to_id[f"dieu_{section.article_number}_khoan_{section.clause_number}"] = section_id
            section_rows.append({
                "section_id": section_id,
                "article_number": section.article_number,
                "clause_number": section.clause_number,
                "point_number": section.point_number,
                "content": section.content,
                "level": section.level,
            })
        return section_rows
