class PostgresBM25:
    def __init__(self, db):
        self._db = db

    def search(self, query: str, top_k: int = 50) -> list:
        rows = self._db.execute_query(
            """
            SELECT
                lc.id,
                lc.chunk_text,
                lc.version_id,
                lc.section_id,
                ld.title,
                ld.document_type,
                ls.article_number,
                ls.clause_number,
                dv.status,
                ts_rank_cd(
                    to_tsvector('simple', lc.chunk_text),
                    plainto_tsquery('simple', %s)
                ) AS score
            FROM public.legal_chunks lc
            JOIN public.document_versions dv ON lc.version_id = dv.id
            JOIN public.legal_documents ld ON dv.document_id = ld.id
            LEFT JOIN public.legal_sections ls ON lc.section_id = ls.id
            WHERE to_tsvector('simple', lc.chunk_text) @@ plainto_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, top_k),
        )

        results = []
        for row in (rows or []):
            results.append({
                "chunk_id": row[0],
                "chunk_text": row[1],
                "content": row[1],
                "version_id": row[2],
                "section_id": row[3],
                "document_title": row[4] or "",
                "document_type": row[5] or "",
                "article_number": row[6] or "",
                "clause_number": row[7] or "",
                "effective_status": row[8] or "active",
                "bm25_score": float(row[9]),
                "score": float(row[9]),
            })
        return results
