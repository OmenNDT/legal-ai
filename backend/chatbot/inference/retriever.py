from dataclasses import dataclass
import psycopg2
from ..data_pipeline.db_loader import DbConfig
from ..data_pipeline.embedder import Embedder

@dataclass
class RetrievedChunk:
    chunk_id: int
    dieu: str
    khoan: str
    diem: str
    full_text: str
    similarity: float
    doc_name: str = ""
    doc_code: str = ""

class Retriever:
    def __init__(self, config: DbConfig, embedder: Embedder, top_k: int = 8, vector_weight: float = 0.50, trigram_weight: float = 0.18, dieu_weight: float = 0.22, keyphrase_weight: float = 0.10, rrf_k: int = 60) -> None:
        self._cfg = config
        self._embedder = embedder
        self._top_k = top_k
        self._vector_weight = vector_weight
        self._trigram_weight = trigram_weight
        self._dieu_weight = dieu_weight
        self._keyphrase_weight = keyphrase_weight
        self._rrf_k = rrf_k

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port,
            dbname = c.db, user = c.user, password = c.password
        )

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        vec = self._embedder.embed_one(question)
        vec_str = "[" + ",".join(map(str, vec)) + "]"
        k = self._top_k
        rrf_k = self._rrf_k
        vw = self._vector_weight
        tw = self._trigram_weight
        dw = self._dieu_weight
        kw = self._keyphrase_weight
        pool = max(k * 30, 200)

        # RRF 4 nguồn:
        #  - vector_ranked:    cosine HNSW trên embedding
        #  - trigram_ranked:   pg_trgm trên content (toàn văn chunk)
        #  - dieu_ranked:      pg_trgm trên dieu (tên Điều)
        #  - keyphrase_ranked: khớp keyphrase đã index (exact + trigram trên phrase)
        # Boost: chunk gốc (khoan='' và diem='') được +0.005 để ưu tiên Điều tổng hợp.
        sql = """
            WITH vector_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank,
                    1 - (embedding <=> %(vec)s::vector) AS vec_score
                FROM law_chunks
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(pool)s
            ),
            trigram_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY similarity(content, %(q)s) DESC) AS rank
                FROM law_chunks
                WHERE content %% %(q)s
                ORDER BY similarity(content, %(q)s) DESC
                LIMIT %(pool)s
            ),
            dieu_ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY similarity(COALESCE(dieu,''), %(q)s) DESC) AS rank
                FROM law_chunks
                WHERE dieu IS NOT NULL AND similarity(dieu, %(q)s) > 0.15
                ORDER BY similarity(dieu, %(q)s) DESC
                LIMIT %(pool)s
            ),
            keyphrase_ranked AS (
                SELECT
                    k.chunk_id AS id,
                    ROW_NUMBER() OVER (ORDER BY MAX(similarity(k.phrase, %(q)s)) DESC) AS rank
                FROM keyphrases k
                WHERE k.phrase ILIKE %(ilike_q)s
                   OR similarity(k.phrase, %(q)s) >= 0.3
                GROUP BY k.chunk_id
                LIMIT %(pool)s
            ),
            rrf AS (
                SELECT
                    COALESCE(v.id, t.id, e.id, kp.id) AS id,
                    COALESCE(%(vw)s / (%(rrf_k)s + v.rank),  0) +
                    COALESCE(%(tw)s / (%(rrf_k)s + t.rank),  0) +
                    COALESCE(%(dw)s / (%(rrf_k)s + e.rank),  0) +
                    COALESCE(%(kw)s / (%(rrf_k)s + kp.rank), 0) AS rrf_score,
                    COALESCE(v.vec_score, 0) AS vec_score
                FROM vector_ranked v
                FULL OUTER JOIN trigram_ranked  t  ON v.id = t.id
                FULL OUTER JOIN dieu_ranked     e  ON COALESCE(v.id, t.id) = e.id
                FULL OUTER JOIN keyphrase_ranked kp ON COALESCE(v.id, t.id, e.id) = kp.id
            )
            SELECT lc.id, lc.dieu, lc.khoan, lc.diem, lc.full_text, r.vec_score,
                   d.doc_name, d.doc_code,
                   r.rrf_score +
                     CASE WHEN COALESCE(lc.khoan,'')='' AND COALESCE(lc.diem,'')='' THEN 0.005 ELSE 0 END
                     AS final_score
            FROM rrf r
            JOIN law_chunks lc ON lc.id = r.id
            LEFT JOIN documents d ON d.id = lc.document_id
            ORDER BY final_score DESC
            LIMIT %(k)s
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, {
                "vec": vec_str,
                "q": question,
                "ilike_q": f"%{question}%",
                "pool": pool,
                "vw": vw,
                "tw": tw,
                "dw": dw,
                "kw": kw,
                "rrf_k": rrf_k,
                "k": k
            })
            return [
                RetrievedChunk(
                    chunk_id = r[0],
                    dieu = r[1] or "",
                    khoan = r[2] or "",
                    diem = r[3] or "",
                    full_text = r[4] or "",
                    similarity = float(r[5]),
                    doc_name = r[6] or "",
                    doc_code = r[7] or ""
                )
                for r in cur.fetchall()
            ]
