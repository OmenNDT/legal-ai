from backend.knowledge.graph_builder import LegalKnowledgeGraph
from backend.search.search_engine import LegalSearchEngine
from backend.common.text_processor import clean_vietnamese

class IndexAndKGBuilder:
    def __init__(self, db, kg_path: str, index_path: str):
        self._db = db
        self._kg_path = kg_path
        self._index_path = index_path

    def build(self) -> None:
        self._build_bm25_index()
        self._build_knowledge_graph()

    def _build_bm25_index(self) -> None:
        rows = self._db.execute_query(
            "SELECT lc.chunk_text, ld.title, ld.document_type "
            "FROM legal_chunks lc "
            "JOIN document_versions dv ON lc.version_id = dv.id "
            "JOIN legal_documents ld ON dv.document_id = ld.id "
            "WHERE dv.status = 'active'"
        )
        if not rows:
            return

        engine = LegalSearchEngine()
        for chunk_text, title, doc_type in rows:
            content = clean_vietnamese(chunk_text or "")
            if content:
                engine.add_document(content, {"title": title or "", "type": doc_type or ""})

        engine.build()
        engine.index.save(self._index_path)

    def _build_knowledge_graph(self) -> None:
        rows = self._db.execute_query(
            "SELECT ld.id, ld.title, ld.document_type, ld.document_number, "
            "       dv.status, dv.effective_date, ld.issuing_body, dv.raw_text "
            "FROM legal_documents ld "
            "JOIN document_versions dv ON dv.document_id = ld.id "
            "WHERE dv.status = 'active'"
        )
        if not rows:
            return

        kg = LegalKnowledgeGraph()
        for doc_id, title, doc_type, doc_number, status, effective_date, issuing_body, raw_text in rows:
            kg.add_document({
                "id": str(doc_id),
                "title": title or "",
                "type": doc_type or "",
                "law_number": doc_number or "",
                "status": status or "active",
                "effective_date": str(effective_date) if effective_date else "",
                "issuer": issuing_body or "",
                "content": (raw_text or "")[:5000],
            })

        kg.save(self._kg_path)
