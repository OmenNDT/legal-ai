import logging
import re

logger = logging.getLogger(__name__)

class DocumentChunker:
    MAX_TOKENS = 512
    OVERLAP_TOKENS = 64
    MIN_CHUNK_TOKENS = 20
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")

    def __init__(self, db):
        self._db = db

    def chunk(self, structured_docs: list) -> list:
        logger.info("[Chunk] Input: %d doc(s)", len(structured_docs))
        all_chunks = []
        for doc in structured_docs:
            before = len(all_chunks)
            all_chunks.extend(self._chunk_doc(doc))
            logger.info("[Chunk] %s → %d chunk(s)", doc.get("filename", "?"), len(all_chunks) - before)
        logger.info("[Chunk] Total: %d chunk(s)", len(all_chunks))
        return all_chunks

    def _chunk_doc(self, doc: dict) -> list:
        version_id = doc["version_id"]
        doc_title = doc.get("document_title", "")
        doc_type = doc.get("document_type", "")
        effective_status = doc.get("effective_status", "active")
        sections = doc.get("sections", [])

        target_sections = self._select_leaf_sections(sections)
        raw_chunks = []
        for section in target_sections:
            raw_chunks.extend(
                self._chunk_section(section, version_id, doc_title, doc_type, effective_status)
            )

        if not raw_chunks:
            text = doc.get("clean_text") or doc.get("raw_text", "")
            if text.strip():
                raw_chunks = self._sliding_window(
                    text, None, version_id, doc_title, doc_type, effective_status, "", "", ""
                )

        return self._save_chunks(raw_chunks)

    def _select_leaf_sections(self, sections: list) -> list:
        if not sections:
            return []
        levels = {s["level"] for s in sections}
        if 3 in levels:
            return [s for s in sections if s["level"] == 3]
        if 2 in levels:
            return [s for s in sections if s["level"] == 2]
        return [s for s in sections if s["level"] == 1]

    def _chunk_section(
        self,
        section: dict,
        version_id: int,
        doc_title: str,
        doc_type: str,
        effective_status: str,
    ) -> list:
        content = section["content"]
        token_count = self._count_tokens(content)
        if token_count <= self.MAX_TOKENS:
            return [{
                "version_id": version_id,
                "section_id": section["section_id"],
                "chunk_text": content,
                "chunk_index": 0,
                "token_count": token_count,
                "article_number": section["article_number"],
                "clause_number": section["clause_number"],
                "point_number": section["point_number"],
                "document_title": doc_title,
                "document_type": doc_type,
                "effective_status": effective_status,
            }]
        return self._sliding_window(
            content,
            section["section_id"],
            version_id,
            doc_title,
            doc_type,
            effective_status,
            section["article_number"],
            section["clause_number"],
            section["point_number"],
        )

    def _sliding_window(
        self,
        text: str,
        section_id,
        version_id: int,
        doc_title: str,
        doc_type: str,
        effective_status: str,
        article: str,
        clause: str,
        point: str,
    ) -> list:
        sentences = [s for s in self._SENTENCE_SPLIT_RE.split(text) if s.strip()]
        chunks = []
        current: list = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            st = self._count_tokens(sentence)
            if current_tokens + st > self.MAX_TOKENS and current:
                chunk_text = " ".join(current)
                if self._count_tokens(chunk_text) >= self.MIN_CHUNK_TOKENS:
                    chunks.append(self._build_chunk(
                        chunk_text, chunk_index, section_id, version_id,
                        doc_title, doc_type, effective_status, article, clause, point,
                    ))
                    chunk_index += 1
                overlap: list = []
                overlap_tokens = 0
                for s in reversed(current):
                    t = self._count_tokens(s)
                    if overlap_tokens + t > self.OVERLAP_TOKENS:
                        break
                    overlap.insert(0, s)
                    overlap_tokens += t
                current = overlap
                current_tokens = overlap_tokens
            current.append(sentence)
            current_tokens += st

        if current:
            chunk_text = " ".join(current)
            if self._count_tokens(chunk_text) >= self.MIN_CHUNK_TOKENS:
                chunks.append(self._build_chunk(
                    chunk_text, chunk_index, section_id, version_id,
                    doc_title, doc_type, effective_status, article, clause, point,
                ))
        return chunks

    def _build_chunk(
        self,
        text: str,
        index: int,
        section_id,
        version_id: int,
        doc_title: str,
        doc_type: str,
        effective_status: str,
        article: str,
        clause: str,
        point: str,
    ) -> dict:
        return {
            "version_id": version_id,
            "section_id": section_id,
            "chunk_text": text,
            "chunk_index": index,
            "token_count": self._count_tokens(text),
            "article_number": article,
            "clause_number": clause,
            "point_number": point,
            "document_title": doc_title,
            "document_type": doc_type,
            "effective_status": effective_status,
        }

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _save_chunks(self, chunks: list) -> list:
        saved = []
        for chunk in chunks:
            try:
                rows = self._db.execute_query(
                    "INSERT INTO legal_chunks "
                    "(version_id, section_id, chunk_text, chunk_index, token_count) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (
                        chunk["version_id"],
                        chunk["section_id"],
                        chunk["chunk_text"],
                        chunk["chunk_index"],
                        chunk["token_count"],
                    ),
                )
                if rows:
                    saved.append({**chunk, "chunk_id": rows[0][0]})
                    logger.debug("[Chunk] Saved chunk_id=%d index=%d tokens=%d", rows[0][0], chunk["chunk_index"], chunk["token_count"])
            except Exception as e:
                logger.error("[Chunk] DB insert FAILED: %s", e, exc_info=True)
        logger.info("[Chunk] Saved %d/%d chunk(s) to DB", len(saved), len(chunks))
        return saved
