from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Chunk:
    chunk_id: int
    content: str
    document_title: str
    document_type: str
    article_number: str
    clause_number: str
    point_number: str
    effective_status: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    version_id: int = 0
    section_id: int = 0

class Retrieval:
    TOP_K = 50

    def __init__(self, hybrid_search, db=None):
        self._hybrid_search = hybrid_search
        self._db = db

    def retrieve(self, structured_query) -> list:
        queries = self._generate_queries(structured_query)
        raw_results = self._hybrid_search.search(queries, n_results=self.TOP_K)
        filtered = self._filter_by_metadata(raw_results, structured_query)
        chunks = [self._to_chunk(r) for r in filtered]
        if self._db and structured_query.query_id:
            self._log_retrieval(chunks, structured_query)
        return chunks

    def _generate_queries(self, structured_query) -> list:
        queries = [structured_query.rewritten_query]
        queries.extend(structured_query.expanded_queries[:3])
        return list(dict.fromkeys(queries))

    def _filter_by_metadata(self, results: list, structured_query) -> list:
        entity_types = {e["entity"] for e in structured_query.entities}
        doc_type_map = {
            "LUAT": "luat",
            "NGHI_DINH": "nghi_dinh",
            "THONG_TU": "thong_tu",
            "QUYET_DINH": "quyet_dinh",
        }
        allowed_types = {doc_type_map[t] for t in entity_types if t in doc_type_map}

        active = []
        expired = []
        for r in results:
            if allowed_types:
                doc_type = r.get("document_type", "").lower().replace(" ", "_")
                if doc_type not in allowed_types:
                    continue
            status = r.get("effective_status", "active")
            if status in ("expired", "replaced"):
                expired.append(r)
            else:
                active.append(r)
        return (active + expired)[:self.TOP_K]

    def _to_chunk(self, raw: dict) -> Chunk:
        return Chunk(
            chunk_id=raw.get("chunk_id", raw.get("id", 0)),
            content=raw.get("chunk_text", raw.get("content", "")),
            document_title=raw.get("document_title", raw.get("title", "")),
            document_type=raw.get("document_type", ""),
            article_number=raw.get("article_number", ""),
            clause_number=raw.get("clause_number", ""),
            point_number=raw.get("point_number", ""),
            effective_status=raw.get("effective_status", "active"),
            bm25_score=raw.get("bm25_score", 0.0),
            vector_score=raw.get("vector_score", 0.0),
            hybrid_score=raw.get("hybrid_score", 0.0),
            version_id=raw.get("version_id", 0),
            section_id=raw.get("section_id", 0),
        )

    def _log_retrieval(self, chunks: list, structured_query):
        for rank, chunk in enumerate(chunks):
            try:
                self._db.execute_query(
                    "INSERT INTO retrieval_logs (query_id, chunk_id, score, rank, retrieval_type) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (structured_query.query_id, chunk.chunk_id, chunk.hybrid_score, rank, "hybrid"),
                    fetch=False,
                )
            except Exception:
                pass
