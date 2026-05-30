"""
Tra cứu theo keyphrase (từ khóa đã index sẵn) cho văn bản luật.

Hỗ trợ 2 chế độ:
  1. Tìm chunk chứa keyphrase khớp chính xác / gần đúng (pg_trgm)
  2. Lọc theo concept_type (định nghĩa, nghĩa vụ, chế tài, ...)

Kết quả trả về list[KeyphraseHit] — tương tự RetrievedChunk nhưng có thêm
trường `matched_phrases` và `concept_types`.

Cách dùng:
    from chatbot.inference.keyphrase_retriever import KeyphraseRetriever
    retriever = KeyphraseRetriever(cfg)
    hits = retriever.search("báo cáo tài chính", doc_code="LKT2015", top_k=10)
    hits = retriever.search_by_concept("che_tai", doc_code="LKT2015")
"""

from dataclasses import dataclass, field
import psycopg2
from ..data_pipeline.db_loader import DbConfig

@dataclass
class KeyphraseHit:
    chunk_id: int
    dieu: str
    khoan: str
    diem: str
    content: str
    doc_name: str
    doc_code: str
    matched_phrases: list[str] = field(default_factory=list)
    concept_types: list[str] = field(default_factory=list)
    score: float = 0.0

class KeyphraseRetriever:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host=c.host, port=c.port, dbname=c.db,
            user=c.user, password=c.password
        )

    # Tra cứu keyphrase (Exact + Fuzzy)
    def search(self, query: str, doc_code: str | None = None, concept_type: str | None = None, top_k: int = 10, sim_threshold: float = 0.3) -> list[KeyphraseHit]:
        """
        Tìm các chunk có keyphrase khớp với query.

        Kết hợp 2 signal:
          - exact/prefix match: ILIKE '%query%' trên phrase
          - fuzzy trigram: similarity(phrase, query) >= sim_threshold
        """
        doc_filter = "AND (d.doc_code = %(doc_code)s OR d.short_code = %(doc_code)s)" if doc_code else ""
        concept_filter = "AND ct.concept_type = %(concept_type)s" if concept_type else ""

        sql = f"""
            WITH matched_kp AS (
                SELECT
                    k.chunk_id,
                    array_agg(DISTINCT k.phrase ORDER BY k.phrase) AS matched_phrases,
                    MAX(similarity(k.phrase, %(query)s)) AS best_sim
                FROM keyphrases k
                WHERE
                    k.phrase ILIKE %(ilike_q)s
                    OR similarity(k.phrase, %(query)s) >= %(sim_threshold)s
                GROUP BY k.chunk_id
            ),
            tagged AS (
                SELECT
                    chunk_id,
                    array_agg(concept_type ORDER BY confidence DESC) AS concept_types
                FROM concept_tags
                GROUP BY chunk_id
            )
            SELECT
                lc.id,
                lc.dieu,
                lc.khoan,
                lc.diem,
                lc.content,
                d.doc_name,
                d.doc_code,
                mk.matched_phrases,
                COALESCE(t.concept_types, ARRAY[]::text[]) AS concept_types,
                mk.best_sim AS score
            FROM matched_kp mk
            JOIN law_chunks lc ON lc.id = mk.chunk_id
            JOIN documents d ON d.id = lc.document_id
            LEFT JOIN tagged t ON t.chunk_id = lc.id
            WHERE 1=1
            {doc_filter}
            {concept_filter}
            ORDER BY mk.best_sim DESC, mk.chunk_id
            LIMIT %(top_k)s
        """

        params: dict = {
            "query": query,
            "ilike_q": f"%{query}%",
            "sim_threshold": sim_threshold,
            "top_k": top_k,
        }
        if doc_code:
            params["doc_code"] = doc_code
        if concept_type:
            params["concept_type"] = concept_type

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            KeyphraseHit(
                chunk_id = r[0],
                dieu = r[1] or "",
                khoan = r[2] or "",
                diem = r[3] or "",
                content = r[4] or "",
                doc_name = r[5] or "",
                doc_code = r[6] or "",
                matched_phrases = list(r[7]) if r[7] else [],
                concept_types = list(r[8]) if r[8] else [],
                score = float(r[9]) if r[9] else 0.0
            )
            for r in rows
        ]
    
    # Tra cứu theo loại khái niệm pháp lý
    def search_by_concept(self, concept_type: str, doc_code: str | None = None, min_confidence: float = 0.3, top_k: int = 20) -> list[KeyphraseHit]:
        """
        Lấy tất cả chunk thuộc một concept_type nhất định.

        Ví dụ:
            retriever.search_by_concept("che_tai", doc_code = "LKT2015")
            → danh sách điều khoản về xử phạt vi phạm kế toán
        """
        doc_filter = "AND (d.doc_code = %(doc_code)s OR d.short_code = %(doc_code)s)" if doc_code else ""

        sql = f"""
            WITH tagged AS (
                SELECT chunk_id, array_agg(concept_type ORDER BY confidence DESC) AS concept_types
                FROM concept_tags
                GROUP BY chunk_id
            ),
            matched_ct AS (
                SELECT chunk_id, confidence
                FROM concept_tags
                WHERE concept_type = %(concept_type)s
                  AND confidence >= %(min_confidence)s
            ),
            kp AS (
                SELECT chunk_id, array_agg(phrase ORDER BY rank) AS phrases
                FROM keyphrases
                GROUP BY chunk_id
            )
            SELECT
                lc.id,
                lc.dieu,
                lc.khoan,
                lc.diem,
                lc.content,
                d.doc_name,
                d.doc_code,
                COALESCE(kp.phrases, ARRAY[]::text[]),
                COALESCE(t.concept_types, ARRAY[]::text[]),
                mc.confidence AS score
            FROM matched_ct mc
            JOIN law_chunks lc ON lc.id = mc.chunk_id
            JOIN documents d ON d.id = lc.document_id
            LEFT JOIN tagged t ON t.chunk_id = lc.id
            LEFT JOIN kp ON kp.chunk_id = lc.id
            WHERE 1=1 {doc_filter}
            ORDER BY mc.confidence DESC, lc.id
            LIMIT %(top_k)s
        """

        params: dict = {
            "concept_type": concept_type,
            "min_confidence": min_confidence,
            "top_k": top_k
        }
        if doc_code:
            params["doc_code"] = doc_code

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            KeyphraseHit(
                chunk_id = r[0],
                dieu = r[1] or "",
                khoan = r[2] or "",
                diem = r[3] or "",
                content = r[4] or "",
                doc_name = r[5] or "",
                doc_code = r[6] or "",
                matched_phrases = list(r[7]) if r[7] else [],
                concept_types = list(r[8]) if r[8] else [],
                score = float(r[9]) if r[9] else 0.0
            )
            for r in rows
        ]

    # Lấy top keyphrase phổ biến nhất trong một văn bản luật
    def get_top_keyphrases(self, doc_code: str, limit: int = 50) -> list[tuple[str, int]]:
        
        # Trả về (phrase, count) — những keyphrase xuất hiện nhiều nhất trong văn bản.
        sql = """
            SELECT k.phrase, COUNT(*) AS cnt
            FROM keyphrases k
            JOIN law_chunks lc ON lc.id = k.chunk_id
            JOIN documents d ON d.id = lc.document_id
            WHERE (d.doc_code = %(doc_code)s OR d.short_code = %(doc_code)s)
            GROUP BY k.phrase
            ORDER BY cnt DESC
            LIMIT %(limit)s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, {"doc_code": doc_code, "limit": limit})
            return [(r[0], r[1]) for r in cur.fetchall()]
